import os
import time
import json
from pathlib import Path
from openai import AzureOpenAI
import logging

# --- Configuration ---
# Ensure these environment variables are set before running the script
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = "https://openai-pinkdotai.openai.azure.com/"
API_VERSION = "2025-01-01-preview" # As per your original script, ensure this is supported
# Define input and output directories
BASE_DATA_DIR = Path("data")
BATCH_INPUT_DIR = BASE_DATA_DIR / "bigger_bachtes"
BATCH_OUTPUT_DIR = BASE_DATA_DIR / "bigger_bachtes_output"

# Batch processing settings
MAX_CONCURRENT_BATCHES = 4 # Adjust as needed, based on your Azure limits (typically 3-5 for some regions/tiers)
POLL_INTERVAL_SECONDS = 30  # How often to check batch statuses

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Validate configuration
if not API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable not set.")
if not AZURE_ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set.")

# Initialize AzureOpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

# --- Helper Functions ---

def generate_output_filename(input_file_path: Path, suffix: str = "") -> str:
    """Generates an output filename based on the input filename."""
    return f"output_{input_file_path.stem}{suffix}{input_file_path.suffix}"

def upload_input_file(file_path: Path):
    """Uploads a file to Azure OpenAI for batch processing."""
    logger.info(f"Uploading input file: {file_path.name}")
    try:
        with file_path.open("rb") as f:
            uploaded_file = client.files.create(file=f, purpose="batch")
        logger.info(f"Successfully uploaded {file_path.name}. Input File ID: {uploaded_file.id}")
        return uploaded_file.id
    except Exception as e:
        logger.error(f"❌ Error uploading {file_path.name}: {e}")
        return None

def create_batch_job(input_file_id: str, input_filename: str):
    """Creates an Azure OpenAI batch job."""
    logger.info(f"Creating batch job for Input File ID: {input_file_id} (from {input_filename})")
    try:
        # The 'endpoint' here refers to the API operation the batch will perform,
        # e.g., "/chat/completions". The model used is specified *within* each request
        # in your .jsonl input file if you need specific models per request.
        # If all requests in the batch use the same model, that model should be associated
        # with the deployment that your Azure OpenAI resource endpoint routes to by default
        # or as specified in the "model" field of each JSON line in the input file.
        # The `metadata` field is for your own tracking.
        batch_job = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/chat/completions", # This is the operation, e.g., chat completions
            completion_window="24h",
            metadata={
                "source_file": input_filename,
                "description": f"Batch processing for {input_filename}"
                # "deployment_name": DEPLOYMENT_NAME # This is often not needed here; model is in input file
            }
        )
        logger.info(f"Batch job created for {input_filename}. Batch ID: {batch_job.id}, Status: {batch_job.status}")
        return batch_job.id
    except Exception as e:
        logger.error(f"❌ Error creating batch job for Input File ID {input_file_id} (from {input_filename}): {e}")
        return None

def download_batch_file_content(file_id: str, save_path: Path):
    """Downloads content of a file (output or error) from Azure OpenAI."""
    if not file_id:
        logger.warning(f"Download requested for a null file_id. Skipping download to {save_path}.")
        return False
    logger.info(f"Attempting to download file ID {file_id} to {save_path}")
    try:
        # Get the content of the file
        file_content_response = client.files.content(file_id)
        # The response is a HttpxBinaryResponseContent, read its content
        content_bytes = file_content_response.read()

        with save_path.open("wb") as f:
            f.write(content_bytes)
        logger.info(f"Successfully downloaded and saved file ID {file_id} to {save_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading file ID {file_id} to {save_path}: {e}")
        try:
            # Try to get more details if it's an API error object
            error_details = client.files.retrieve(file_id)
            logger.error(f"File details for {file_id}: {error_details}")
        except Exception as detail_e:
            logger.error(f"Could not retrieve details for file ID {file_id}: {detail_e}")
        return False


# --- Main Process ---
def main():
    logger.info("--- Starting Robust Batch Processing Script ---")

    # 1. Discover input files and filter out already processed ones
    all_input_files = sorted(list(BATCH_INPUT_DIR.glob("*.jsonl")))
    files_to_process_paths = []
    for file_path in all_input_files:
        # Check if a corresponding output file already exists
        # This simple check assumes output files are named based on input files.
        # A more robust system might use a database or status file.
        expected_output_name = generate_output_filename(file_path)
        if (BATCH_OUTPUT_DIR / expected_output_name).exists():
            logger.info(f"Skipping '{file_path.name}': Output file '{expected_output_name}' already exists.")
            continue
        files_to_process_paths.append(file_path)

    if not files_to_process_paths:
        logger.info("No new input files to process. Exiting.")
        return

    logger.info(f"Found {len(files_to_process_paths)} new input files to process.")

    # Stores info about batches: {batch_id: {'input_file_path': Path, 'status': str, 'input_file_id': str}}
    active_batches = {}
    # Queue of input file paths to process
    pending_files_queue = list(files_to_process_paths) # Make a mutable copy

    completed_batch_count = 0
    total_files_to_process = len(pending_files_queue)

    try:
        while pending_files_queue or active_batches:
            # --- Phase 1: Submit new batch jobs if capacity allows ---
            while len(active_batches) < MAX_CONCURRENT_BATCHES and pending_files_queue:
                input_file_path = pending_files_queue.pop(0) # Get next file from the front
                logger.info(f"\n--- Processing next file: {input_file_path.name} ---")

                # Check again if output exists, in case it was created by another process
                # or a previous run that was interrupted after file creation but before tracking.
                expected_output_name = generate_output_filename(input_file_path)
                if (BATCH_OUTPUT_DIR / expected_output_name).exists():
                    logger.info(f"Skipping '{input_file_path.name}' again: Output file '{expected_output_name}' now exists.")
                    total_files_to_process -=1 # Adjust total if skipped now
                    if total_files_to_process == 0 and not active_batches: # All files processed or skipped
                         break
                    continue


                input_file_id = upload_input_file(input_file_path)
                if input_file_id:
                    batch_id = create_batch_job(input_file_id, input_file_path.name)
                    if batch_id:
                        active_batches[batch_id] = {
                            'input_file_path': input_file_path,
                            'status': 'submitted', # Initial status after creation attempt
                            'input_file_id': input_file_id,
                            'output_file_id': None,
                            'error_file_id': None
                        }
                        logger.info(f"Batch job {batch_id} for {input_file_path.name} added to active pool.")
                    else:
                        logger.warning(f"Failed to create batch job for {input_file_path.name}. Will not be tracked as active.")
                        # Optionally, move this file to an error/retry directory
                else:
                    logger.warning(f"Failed to upload {input_file_path.name}. Will not create batch job.")
                    # Optionally, move this file to an error/retry directory
                logger.info("-" * 30)

            if not pending_files_queue and not active_batches and total_files_to_process == 0 :
                logger.info("All files processed or skipped, and no active batches. Initial loop check.")
                break


            # --- Phase 2: Monitor active batch jobs ---
            if active_batches:
                logger.info(f"\n--- Monitoring {len(active_batches)} active batch jobs ---")
                # Create a copy of keys to iterate over, as we might modify the dict
                batch_ids_to_check = list(active_batches.keys())
                for batch_id in batch_ids_to_check:
                    if batch_id not in active_batches: # Might have been removed by a previous iteration
                        continue

                    batch_info = active_batches[batch_id]
                    logger.info(f"Checking status for Batch ID: {batch_id} (File: {batch_info['input_file_path'].name})")
                    try:
                        retrieved_batch = client.batches.retrieve(batch_id)
                        current_status = retrieved_batch.status
                        active_batches[batch_id]['status'] = current_status
                        active_batches[batch_id]['output_file_id'] = retrieved_batch.output_file_id
                        active_batches[batch_id]['error_file_id'] = retrieved_batch.error_file_id

                        logger.info(f"Batch ID: {batch_id}, Status: {current_status}, Output File ID: {retrieved_batch.output_file_id}, Error File ID: {retrieved_batch.error_file_id}")

                        if current_status == 'completed':
                            logger.info(f"✅ Batch {batch_id} for {batch_info['input_file_path'].name} completed!")
                            output_file_id = retrieved_batch.output_file_id
                            if output_file_id:
                                output_filename = generate_output_filename(batch_info['input_file_path'])
                                save_path = BATCH_OUTPUT_DIR / output_filename
                                download_batch_file_content(output_file_id, save_path)
                            else:
                                logger.warning(f"Batch {batch_id} completed but no output_file_id was provided.")
                            del active_batches[batch_id]
                            completed_batch_count += 1
                            logger.info(f"Removed {batch_id} from active pool. Completed: {completed_batch_count}/{total_files_to_process}")


                        elif current_status in ['failed', 'cancelled', 'expired']:
                            logger.error(f"💔 Batch {batch_id} for {batch_info['input_file_path'].name} status: {current_status}.")
                            error_file_id = retrieved_batch.error_file_id
                            if error_file_id:
                                error_filename = generate_output_filename(batch_info['input_file_path'], suffix="_error")
                                error_save_path = BATCH_OUTPUT_DIR / error_filename
                                logger.info(f"Attempting to download error file for batch {batch_id} to {error_save_path}")
                                download_batch_file_content(error_file_id, error_save_path)
                            else:
                                logger.warning(f"Batch {batch_id} {current_status} but no error_file_id was provided.")
                            del active_batches[batch_id]
                            completed_batch_count += 1 # Count as processed for loop termination
                            logger.info(f"Removed {batch_id} from active pool due to failure/cancellation. Processed: {completed_batch_count}/{total_files_to_process}")

                        elif current_status in ['validating', 'in_progress', 'queued']:
                            logger.info(f"Batch {batch_id} is still {current_status}. Will check again later.")
                        else:
                            logger.warning(f"Batch {batch_id} has an unexpected status: {current_status}")

                    except Exception as e:
                        logger.error(f"❌ Error retrieving status for Batch ID {batch_id}: {e}")
                        # Decide if you want to remove it from active_batches or retry N times
                        # For simplicity here, we'll leave it for the next poll cycle unless it's a fatal error.
                        # If the error indicates the batch_id is invalid, then remove it.
                        if "No batch found with id" in str(e): # Example error string, adjust if necessary
                            logger.error(f"Batch ID {batch_id} seems invalid. Removing from active pool.")
                            del active_batches[batch_id]
                            completed_batch_count += 1 # Count as processed
                            logger.info(f"Processed count updated: {completed_batch_count}/{total_files_to_process}")


            # If no files are pending and no batches are active, we are done.
            if not pending_files_queue and not active_batches:
                logger.info("All tasks processed and active batches cleared.")
                break # Exit main loop

            logger.info(f"Waiting for {POLL_INTERVAL_SECONDS} seconds before next cycle. "
                        f"Pending files: {len(pending_files_queue)}, Active batches: {len(active_batches)}")
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("\n🛑 Process interrupted by user. Exiting gracefully...")
    finally:
        logger.info(f"--- Batch Processing Script Finished ---")
        logger.info(f"Total files initially targeted: {len(all_input_files)}")
        logger.info(f"Total files attempted for processing in this run: {total_files_to_process}")
        logger.info(f"Batches successfully completed and downloaded: {completed_batch_count} (this may include jobs marked as failed/cancelled if their error files were processed)")
        if active_batches:
            logger.warning(f"There are still {len(active_batches)} batches that were active upon exit:")
            for batch_id, info in active_batches.items():
                logger.warning(f"  - Batch ID: {batch_id}, File: {info['input_file_path'].name}, Status: {info['status']}")

if __name__ == "__main__":
    # Example: Create some dummy input files if they don't exist
    # In a real scenario, these files would already be prepared.
    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()