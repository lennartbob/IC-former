import os
import time
from pathlib import Path
from openai import AzureOpenAI

# --- Configuration ---
# Ensure these environment variables are set before running the script
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-pinkdotai.openai.azure.com/")
API_VERSION = "2025-01-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-2"  # Make sure this is a Global-Batch deployment

# Define input and output directories
BATCH_INPUT_DIR = Path("data/question_batches")
BATCH_OUTPUT_DIR = Path("data/question_batches_output")

# Ensure output directory exists
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
def upload_file(file_path: Path):
    """Uploads a file to Azure OpenAI for batch processing."""
    print(f"Uploading file: {file_path.name}")
    try:
        with file_path.open("rb") as f:
            uploaded_file = client.files.create(file=f, purpose="batch")
        print(f"Uploaded {file_path.name}. File ID: {uploaded_file.id}")
        return uploaded_file.id
    except Exception as e:
        print(f"❌ Error uploading {file_path.name}: {e}")
        return None

def create_batch_job(file_id: str):
    """Creates an Azure OpenAI batch job."""
    print(f"Creating batch job for file ID: {file_id}")
    try:
        response = client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h",
            metadata={"deployment_name": DEPLOYMENT_NAME} # Optional: Add metadata for context
        )
        print(f"Batch job created. Batch ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"❌ Error creating batch job for file ID {file_id}: {e}")
        return None

def download_file(file_id: str, save_path: Path):
    """Downloads a file from Azure OpenAI."""
    print(f"Downloading file ID {file_id} to {save_path}")
    try:
        content = client.files.download(file_id)
        # Ensure content is read as bytes
        if isinstance(content, bytes):
            with open(save_path, "wb") as f:
                f.write(content)
        else: # Handle cases where content might be a stream or other type
            with open(save_path, "wb") as f:
                for chunk in content.iter_bytes(): # Assuming content has iter_bytes for streams
                    f.write(chunk)
        print(f"Successfully downloaded file ID {file_id}.")
    except Exception as e:
        print(f"❌ Error downloading file ID {file_id}: {e}")

# --- Main Process ---
def main():
    """
    Main function to orchestrate concurrent batch processing.
    1. Uploads all files and creates batch jobs.
    2. Monitors all active batch jobs concurrently.
    3. Downloads results for completed jobs.
    """
    print("--- Starting Concurrent Batch Processing ---")

    # List to hold information about active batches
    # Each item: {'file_path': Path, 'batch_id': str, 'status': str}
    active_batches = []
    
    # --- Phase 1: Upload files and create all batch jobs ---
    print("\n--- Phase 1: Uploading files and creating batch jobs ---")
    for file_path in BATCH_INPUT_DIR.glob("*.jsonl"):
        print(f"Initiating processing for: {file_path.name}")
        file_id = upload_file(file_path)
        if file_id:
            batch_id = create_batch_job(file_id)
            if batch_id:
                active_batches.append({
                    'file_path': file_path,
                    'batch_id': batch_id,
                    'status': 'validating' # Initial status
                })
        print("-" * 30) # Separator for clarity

    if not active_batches:
        print("No batch jobs were initiated. Exiting.")
        return

    print(f"\nSuccessfully initiated {len(active_batches)} batch jobs.")
    print("--- Phase 2: Monitoring batch job completion ---")

    # --- Phase 2: Monitor all batch jobs concurrently ---
    polling_interval_seconds = 30 # How often to poll the API for updates
    
    while active_batches:
        print(f"\nPolling batch statuses. Remaining active jobs: {len(active_batches)}")
        
        # Create a list to store jobs that have completed/failed/cancelled in this cycle
        completed_this_cycle = []

        for batch_info in list(active_batches): # Iterate over a copy to allow modification
            batch_id = batch_info['batch_id']
            file_path = batch_info['file_path']
            
            try:
                batch = client.batches.retrieve(batch_id)
                current_status = batch.status
                
                if current_status != batch_info['status']:
                    print(f"Batch {batch_id} ({file_path.name}) status changed: {batch_info['status']} -> {current_status}")
                    batch_info['status'] = current_status # Update status in our local list

                if current_status in {"completed", "failed", "cancelled"}:
                    print(f"Batch {batch_id} ({file_path.name}) finished with status: {current_status}")
                    
                    output_base = BATCH_OUTPUT_DIR / file_path.stem
                    output_base.mkdir(exist_ok=True) # Ensure output directory exists for this batch

                    if batch.output_file_id:
                        download_file(batch.output_file_id, output_base / "output.jsonl")
                    else:
                        print(f"No output file for batch {batch_id}.")
                    
                    if batch.error_file_id:
                        download_file(batch.error_file_id, output_base / "errors.jsonl")
                    else:
                        print(f"No error file for batch {batch_id}.")
                    
                    completed_this_cycle.append(batch_info)
                else:
                    print(f"Batch {batch_id} ({file_path.name}) status: {current_status} - Waiting...")

            except Exception as e:
                print(f"❌ Error retrieving status for batch {batch_id} ({file_path.name}): {e}")
                # Consider marking this batch as problematic and removing it if persistent errors
                # For now, we'll let it stay in active_batches to retry
                
        # Remove completed/failed/cancelled batches from the active list
        for finished_batch in completed_this_cycle:
            active_batches.remove(finished_batch)
            
        if active_batches: # Only sleep if there are still active batches
            print(f"Sleeping for {polling_interval_seconds} seconds before next poll...")
            time.sleep(polling_interval_seconds)

    print("\n✅ All batch jobs processed (completed, failed, or cancelled).")

if __name__ == "__main__":
    main()