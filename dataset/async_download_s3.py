import os
import requests # Still used for the original structure, will be replaced by aiohttp for async
import aiohttp # For asynchronous HTTP requests
import asyncio
import zipfile
import io
import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm
import random
import time
import shutil
import json
from langdetect import detect, DetectorFactory
import concurrent.futures

# Ensure reproducibility for langdetect if needed
DetectorFactory.seed = 0

# --- Configuration ---
BASE_URL = "https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/"
TARGET_PDF_COUNT = 50000
MIN_TOKENS = 1500
MAX_TOKENS = 50_000

OUTPUT_DIR = "data"
TEMP_ZIP_DOWNLOAD_DIR = "temp_zip_processing_async" # Changed to avoid conflict if old exists
JSON_OUTPUT_FILENAME = "collected_pdf_texts_async.json"

TOTAL_ZIP_FILES = 7933 # 0 to 7932

# --- Concurrency Configuration ---
NUM_CONCURRENT_PIPELINES = 3  # Max number of ZIPs being downloaded/processed concurrently
MAX_PROCESSING_WORKERS = os.cpu_count() or 2 # Number of threads for CPU-bound PDF processing

db_path = os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME)

# --- Global Shared State (managed by main orchestrator) ---
collected_data_for_json = []
already_found_pdf_names = set()
valid_pdfs_found_count = 0
stop_all_processing = False # Flag to signal graceful shutdown

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_ZIP_DOWNLOAD_DIR, exist_ok=True)

# Initialize tiktoken tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("Falling back to p50k_base tokenizer for tiktoken.")
    tokenizer = tiktoken.get_encoding("p50k_base")

def load_existing_data():
    global valid_pdfs_found_count, collected_data_for_json, already_found_pdf_names
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                collected_data_for_json.extend(existing_data)
                for item in existing_data:
                    already_found_pdf_names.add(item["filename"])
            valid_pdfs_found_count = len(already_found_pdf_names)
            print(f"Loaded {valid_pdfs_found_count} existing PDF entries from {db_path}.")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not load existing JSON database {db_path}: {e}. Starting fresh.")
            valid_pdfs_found_count = 0 # Ensure it's reset
    else:
        print(f"No existing JSON database found at {db_path}. Starting fresh.")
        valid_pdfs_found_count = 0 # Ensure it's reset

def get_zip_url_and_local_path(zip_num_int):
    zip_filename_short = f"{zip_num_int:04d}.zip"
    zip_group_start = (zip_num_int // 1000) * 1000
    zip_group_end = zip_group_start + 999
    zip_subdir_path = f"zipfiles/{zip_group_start:04d}-{zip_group_end:04d}/"
    full_zip_url = f"{BASE_URL}{zip_subdir_path}{zip_filename_short}"
    local_download_path = os.path.join(TEMP_ZIP_DOWNLOAD_DIR, zip_filename_short)
    return full_zip_url, local_download_path

async def download_zip_async(session: aiohttp.ClientSession, url: str, local_path: str):
    # print(f"Starting download: {url}")
    async with session.get(url) as response:
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(local_path, 'wb') as f_zip:
            async for chunk in response.content.iter_chunked(8192 * 2):
                f_zip.write(chunk)
    # print(f"Finished download: {url} to {local_path}")

def extract_text_and_count_tokens_from_pdf_data(pdf_data):
    full_text = ""
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        doc.close()
    except Exception: # More specific fitz/PyMuPDF exceptions could be caught
        return None, 0
    if not full_text.strip():
        return "", 0
    tokens = tokenizer.encode(full_text) # Uses global tokenizer
    return full_text, len(tokens)

def detect_language_sync(text): # Renamed to avoid conflict if original is kept
    try:
        if len(text) > 50:
            return detect(text)
        else:
            return "too_short_for_detection"
    except Exception:
        return "unknown"

def process_zip_file_sync(local_zip_filepath: str, current_min_tokens: int, current_max_tokens: int):
    """
    Synchronous function to process a single ZIP file.
    Opens ZIP, extracts PDFs, processes them, and returns data for valid PDFs.
    This function is run in a ThreadPoolExecutor.
    """
    # print(f"SYNC: Starting processing for {local_zip_filepath}")
    found_pdfs_data_in_zip = []
    try:
        with zipfile.ZipFile(local_zip_filepath, 'r') as zf:
            pdf_filenames_in_zip = [
                member.filename for member in zf.infolist()
                if not member.is_dir() and member.filename.lower().endswith('.pdf')
            ]
            for pdf_name_in_zip in pdf_filenames_in_zip:
                # The check for already_found_pdf_names and TARGET_PDF_COUNT
                # will happen in the main async loop to ensure serial access.
                try:
                    pdf_bytes = zf.read(pdf_name_in_zip)
                except Exception:
                    # print(f"SYNC: Error reading {pdf_name_in_zip} from {local_zip_filepath}")
                    continue

                extracted_text, token_count = extract_text_and_count_tokens_from_pdf_data(pdf_bytes)

                if extracted_text is None: # fitz error
                    continue

                if current_min_tokens <= token_count <= current_max_tokens:
                    language = detect_language_sync(extracted_text)
                    found_pdfs_data_in_zip.append({
                        "filename": pdf_name_in_zip,
                        "text": extracted_text,
                        "token_count": token_count,
                        "language": language
                    })
                    # print(f"SYNC: Valid PDF found in {local_zip_filepath}: {pdf_name_in_zip}, Tokens: {token_count}")
    except zipfile.BadZipFile:
        print(f"SYNC: Bad ZIP file: {local_zip_filepath}")
        # This error will be caught by the caller of run_in_executor
        raise # Re-raise to be caught by manage_one_zip
    except Exception as e_proc_zip:
        print(f"SYNC: General error processing zip {local_zip_filepath}: {e_proc_zip}")
        raise # Re-raise
    
    # print(f"SYNC: Finished processing for {local_zip_filepath}, found {len(found_pdfs_data_in_zip)} valid PDFs in it.")
    return found_pdfs_data_in_zip


async def manage_one_zip(
    session: aiohttp.ClientSession, 
    executor: concurrent.futures.ThreadPoolExecutor, 
    zip_idx: int
):
    """
    Manages the lifecycle of a single ZIP file: download, process, return results.
    Returns: (zip_idx, list_of_pdf_data_from_zip, error_occurred_flag, local_zip_filepath_to_delete)
    """
    global MIN_TOKENS, MAX_TOKENS # Access global config
    zip_url, local_zip_filepath = get_zip_url_and_local_path(zip_idx)
    
    try:
        # Ensure the directory for this specific ZIP exists, useful if zips are grouped by subdirs in TEMP_ZIP_DOWNLOAD_DIR
        os.makedirs(os.path.dirname(local_zip_filepath), exist_ok=True)
        await download_zip_async(session, zip_url, local_zip_filepath)
    except Exception as e_download:
        # print(f"ASYNC MANAGE: Download error for ZIP {zip_idx:04d} ({zip_url}): {e_download}")
        if os.path.exists(local_zip_filepath): # Clean up partial download
            try: os.remove(local_zip_filepath)
            except OSError: pass
        return zip_idx, None, True, None # error, no path to delete as it was handled or never fully created

    loop = asyncio.get_event_loop()
    try:
        # print(f"ASYNC MANAGE: Submitting ZIP {zip_idx:04d} ({local_zip_filepath}) for sync processing.")
        pdf_results = await loop.run_in_executor(
            executor,
            process_zip_file_sync, # Synchronous function
            local_zip_filepath,
            MIN_TOKENS, # Pass current config values
            MAX_TOKENS
        )
        # print(f"ASYNC MANAGE: Processing complete for ZIP {zip_idx:04d}.")
        return zip_idx, pdf_results, False, local_zip_filepath # no error, path to delete
    except Exception as e_proc:
        # print(f"ASYNC MANAGE: Sync processing error for ZIP {zip_idx:04d} ({local_zip_filepath}): {e_proc}")
        # The local_zip_filepath should still be returned for deletion
        return zip_idx, None, True, local_zip_filepath # error, path to delete


async def main_orchestrator():
    global valid_pdfs_found_count, stop_all_processing, collected_data_for_json, already_found_pdf_names

    load_existing_data() # Load data at the beginning

    zip_file_indices = list(range(TOTAL_ZIP_FILES))
    # random.shuffle(zip_file_indices) # Uncomment to process in random order

    # Filter out zips if all their potential PDFs are already found (advanced, skip for now for simplicity)
    # This would require knowing which PDFs are in which ZIP beforehand.

    zip_indices_iter = iter(zip_file_indices)

    pbar_valid_pdfs = tqdm(initial=valid_pdfs_found_count, total=TARGET_PDF_COUNT, desc="Valid PDFs Found", unit="PDF")
    pbar_zips = tqdm(total=len(zip_file_indices), desc="Processing ZIPs", unit="ZIP")


    async with aiohttp.ClientSession() as session:
        # Use a context manager for the executor for proper shutdown
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCESSING_WORKERS) as executor:
            active_tasks = set()

            # Initial fill of pipeline tasks
            for _ in range(NUM_CONCURRENT_PIPELINES):
                if stop_all_processing or valid_pdfs_found_count >= TARGET_PDF_COUNT:
                    break
                try:
                    zip_idx = next(zip_indices_iter)
                    task = asyncio.create_task(manage_one_zip(session, executor, zip_idx))
                    active_tasks.add(task)
                except StopIteration:
                    break # No more zips to schedule

            while active_tasks:
                if stop_all_processing and all(not t.done() for t in active_tasks): # If stopping, wait for tasks
                     # Check if we need to break early or let tasks finish
                    pass

                done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                active_tasks = pending # Update active_tasks to only pending ones

                for task in done:
                    try:
                        # result from manage_one_zip:
                        # (zip_idx_processed, pdf_results_from_zip, error_flag, path_to_delete)
                        zip_idx_processed, pdf_results_from_zip, error_flag, path_to_delete = task.result()

                        pbar_zips.update(1)
                        pbar_zips.set_postfix_str(f"ZIP {zip_idx_processed:04d}")

                        if error_flag:
                            print(f"INFO: ZIP {zip_idx_processed:04d} encountered an error during its pipeline.")
                            # Error messages should have been printed within manage_one_zip or process_zip_file_sync
                        
                        if pdf_results_from_zip:
                            for pdf_data in pdf_results_from_zip:
                                if stop_all_processing or valid_pdfs_found_count >= TARGET_PDF_COUNT:
                                    break 
                                
                                # CRITICAL SECTION: Modifying shared data
                                if pdf_data["filename"] not in already_found_pdf_names:
                                    collected_data_for_json.append(pdf_data)
                                    already_found_pdf_names.add(pdf_data["filename"])
                                    valid_pdfs_found_count += 1
                                    pbar_valid_pdfs.update(1)
                                    
                                    if valid_pdfs_found_count >= TARGET_PDF_COUNT:
                                        print(f"\nTarget of {TARGET_PDF_COUNT} valid PDFs reached.")
                                        stop_all_processing = True # Signal to stop scheduling new tasks
                                        # Don't break here, let other tasks in 'done' set complete their processing
                                        # and then new tasks won't be scheduled.
                                else:
                                    # Optional: log if a PDF was processed but already found
                                    # print(f"INFO: PDF {pdf_data['filename']} from ZIP {zip_idx_processed} already in database, skipped adding.")
                                    pass
                        
                        if path_to_delete and os.path.exists(path_to_delete):
                            try:
                                os.remove(path_to_delete)
                            except OSError as e_del:
                                print(f"Error deleting ZIP {path_to_delete} (from ZIP index {zip_idx_processed}): {e_del}")
                        
                    except Exception as e_task_main: # Exception from task itself (e.g. cancellation, or unhandled in manage_one_zip)
                        # This should ideally not happen if manage_one_zip is robust
                        print(f"Orchestrator error processing a task's result: {e_task_main}")
                        # Try to find out which zip_idx it was, if possible (hard if task._coro is complex)


                    # If not stopping and more zips available, schedule a new one to keep pipeline full
                    if not stop_all_processing and valid_pdfs_found_count < TARGET_PDF_COUNT:
                        try:
                            next_zip_to_schedule = next(zip_indices_iter)
                            new_task = asyncio.create_task(manage_one_zip(session, executor, next_zip_to_schedule))
                            active_tasks.add(new_task)
                        except StopIteration:
                            # No more zips to schedule, loop will continue until active_tasks is empty
                            pass
                
                # If stop_all_processing is true and all tasks in the current 'done' batch are processed,
                # and no new tasks were added, active_tasks might become empty, or loop continues until it is.
                if stop_all_processing and not active_tasks:
                    break # All inflight tasks finished after stop signal

    pbar_valid_pdfs.close()
    pbar_zips.close()

    # --- Save the collected data to JSON ---
    if collected_data_for_json:
        print(f"\nSaving collected text data for {len(collected_data_for_json)} PDFs to {db_path}...")
        try:
            collected_data_for_json.sort(key=lambda x: x['filename'])
            with open(db_path, 'w', encoding='utf-8') as f_json:
                json.dump(collected_data_for_json, f_json, ensure_ascii=False, indent=2)
            print("JSON data saved successfully.")
        except IOError as e_json_io:
            print(f"Error saving JSON data: {e_json_io}")
    else:
        print("\nNo new valid PDFs were collected (or previously existed and reloaded) to save to JSON.")

    print(f"\n--- Processing Complete ---")
    print(f"Collected {valid_pdfs_found_count} PDFs meeting the criteria ({MIN_TOKENS}-{MAX_TOKENS} tokens).")
    print(f"JSON data saved in: {os.path.abspath(db_path)}")

    if valid_pdfs_found_count < TARGET_PDF_COUNT and not stop_all_processing: # Ran out of zips
        print(f"Warning: Only found {valid_pdfs_found_count} PDFs after checking all available ZIPs, but targeted {TARGET_PDF_COUNT}.")
    elif valid_pdfs_found_count < TARGET_PDF_COUNT and stop_all_processing: # Target not met but stopped (should be target met)
        # This case should ideally mean target was met, or an external interruption occurred.
        # If stop_all_processing is True because target was met, this message is slightly off.
        # The print for "Target ... reached" already covers it.
        pass


    # --- Cleanup temporary ZIP download directory ---
    try:
        if os.path.exists(TEMP_ZIP_DOWNLOAD_DIR): # Check if it exists before trying to list or remove
            if not os.listdir(TEMP_ZIP_DOWNLOAD_DIR): # Check if empty
                shutil.rmtree(TEMP_ZIP_DOWNLOAD_DIR)
                print(f"Removed empty temporary ZIP directory: {TEMP_ZIP_DOWNLOAD_DIR}")
            else:
                print(f"Temporary ZIP directory {TEMP_ZIP_DOWNLOAD_DIR} is not empty. Manual check may be needed.")
    except Exception as e_cleanup:
        print(f"Error during final cleanup of {TEMP_ZIP_DOWNLOAD_DIR}: {e_cleanup}")


if __name__ == "__main__":
    asyncio.run(main_orchestrator())