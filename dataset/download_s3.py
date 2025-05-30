import os
import requests
import zipfile
import io
import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm
import random
import time
import shutil
import json
from langdetect import detect, DetectorFactory, LangDetectException # Import LangDetectException
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Ensure reproducibility for langdetect if needed
DetectorFactory.seed = 0

# --- Configuration ---
BASE_URL = "https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/"
TARGET_PDF_COUNT = 50000
MIN_TOKENS = 1500
MAX_TOKENS = 50_000

OUTPUT_DIR = "processed_pdf_data" # Renamed to reflect its new purpose (only JSON)
TEMP_ZIP_DOWNLOAD_DIR = "temp_zip_processing"
JSON_OUTPUT_FILENAME = "collected_pdf_texts.json"

TOTAL_ZIP_FILES = 7933

db_path = os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME)

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_ZIP_DOWNLOAD_DIR, exist_ok=True)

# Initialize tiktoken tokenizer globally in main process.
# Each child process will get its own copy if multiprocessing is used, which is fine as it's cached.
try:
    # This global tokenizer is mainly for the main process; workers will initialize their own.
    global_tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("Falling back to p50k_base tokenizer for tiktoken.")
    global_tokenizer = tiktoken.get_encoding("p50k_base")

# Load existing data and populate already_found_pdf_names set
collected_data_for_json = []
already_found_pdf_names = set()
if os.path.exists(db_path):
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            collected_data_for_json.extend(existing_data)
            for item in existing_data:
                already_found_pdf_names.add(item["filename"])
        print(f"Loaded {len(existing_data)} existing PDF entries from {db_path}.")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Could not load existing JSON database {db_path}: {e}. Starting fresh.")
else:
    print(f"No existing JSON database found at {db_path}. Starting fresh.")

initial_valid_pdfs_count = len(already_found_pdf_names)
valid_pdfs_found_count = initial_valid_pdfs_count

def get_zip_url_and_local_path(zip_num_int):
    """
    Generates the URL for a given ZIP number and its intended local download path.
    """
    zip_filename_short = f"{zip_num_int:04d}.zip"
    zip_group_start = (zip_num_int // 1000) * 1000
    zip_group_end = zip_group_start + 999
    zip_subdir_path = f"zipfiles/{zip_group_start:04d}-{zip_group_end:04d}/"
    full_zip_url = f"{BASE_URL}{zip_subdir_path}{zip_filename_short}"
    local_download_path = os.path.join(TEMP_ZIP_DOWNLOAD_DIR, zip_filename_short)
    return full_zip_url, local_download_path

def extract_text_and_count_tokens_from_pdf_data_worker(pdf_data):
    """
    Worker function for PDF text extraction and token counting.
    Runs in a separate process.
    """
    full_text = ""
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        doc.close()
    except Exception: # Catch all exceptions during PDF processing (e.g., malformed PDF)
        return None, 0 # Indicate error

    if not full_text.strip():
        return "", 0 # No text extracted

    # Each process needs its own tokenizer for correctness, but tiktoken caches efficiently
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = tiktoken.get_encoding("p50k_base")

    tokens = tokenizer.encode(full_text)
    return full_text, len(tokens)

def detect_language_worker(text):
    """
    Worker function for language detection.
    Runs in a separate process.
    """
    try:
        if len(text) > 50: # Only try to detect if enough text is present
            return detect(text)
        else:
            return "too_short_for_detection"
    except LangDetectException: # Specific exception for langdetect when it cannot detect
        return "unknown"
    except Exception: # Catch other potential errors during detection
        return "unknown"

def process_single_pdf_task(pdf_name_in_zip, pdf_bytes):
    """
    A single task for ProcessPoolExecutor: extract text, count tokens, detect language.
    Returns a dictionary of results or None if the PDF doesn't meet criteria or has an error.
    """
    extracted_text, token_count = extract_text_and_count_tokens_from_pdf_data_worker(pdf_bytes)

    if extracted_text is None: # fitz processing error
        return None

    if MIN_TOKENS <= token_count <= MAX_TOKENS:
        language = detect_language_worker(extracted_text)
        return {
            "filename": pdf_name_in_zip,
            "text": extracted_text,
            "token_count": token_count,
            "language": language
        }
    return None

def download_and_extract_zip_task(zip_idx):
    """
    A single task for ThreadPoolExecutor: download a ZIP and extract its PDF bytes.
    Returns a list of (filename, pdf_bytes) tuples for all PDFs in the ZIP.
    Handles network errors and retries.
    """
    zip_url, local_zip_filepath = get_zip_url_and_local_path(zip_idx)
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            # Increased timeout to handle large ZIPs or slow connections, adjusted chunk size
            response = requests.get(zip_url, stream=True, timeout=60) 
            response.raise_for_status()

            # Store to disk temporarily for robust handling with zipfile, larger chunk size
            with open(local_zip_filepath, 'wb') as f_zip:
                for chunk in response.iter_content(chunk_size=8192 * 8): 
                    f_zip.write(chunk)

            pdf_data_list = []
            with zipfile.ZipFile(local_zip_filepath, 'r') as zf:
                for member in zf.infolist():
                    if not member.is_dir() and member.filename.lower().endswith('.pdf'):
                        try:
                            pdf_bytes = zf.read(member.filename)
                            pdf_data_list.append((member.filename, pdf_bytes))
                        except Exception: # Catch errors reading individual PDFs from ZIP
                            pass # Skip problematic PDFs within a ZIP

            return pdf_data_list # Return the list of (filename, bytes) tuples

        except requests.exceptions.RequestException as e_req:
            retries += 1
            # print(f"Download/Network error for {zip_url}: {e_req}. Retrying ({retries}/{max_retries})...")
            time.sleep(2 * retries) # Exponential backoff for retries
        except zipfile.BadZipFile:
            # print(f"Corrupt ZIP file downloaded: {local_zip_filepath}. Skipping after {retries+1} retries.")
            return [] # Return empty list if ZIP is bad
        finally:
            if os.path.exists(local_zip_filepath):
                try:
                    os.remove(local_zip_filepath) # Clean up temporary ZIP immediately
                except OSError:
                    pass # Ignore if file cannot be deleted (e.g., already gone)
    
    # print(f"Failed to download and process {zip_url} after {max_retries} retries. Skipping.")
    return [] # Failed to download after all retries

# --- Main Download and Processing Logic ---
zip_file_indices = list(range(TOTAL_ZIP_FILES))
# random.shuffle(zip_file_indices) # Uncomment to process ZIPs in random order for more diverse sample

# Progress bar for overall valid PDFs found
pbar_valid_pdfs = tqdm(initial=initial_valid_pdfs_count, total=TARGET_PDF_COUNT, desc="Valid PDFs Found", unit="PDF")
# Progress bar for ZIPs processed (tracks ZIPs whose download/extraction is finished)
pbar_zips = tqdm(total=len(zip_file_indices), desc="Processing ZIPs", unit="ZIP")


# Adjust these values based on your server's resources and network
NUM_DOWNLOAD_WORKERS = min(16, TOTAL_ZIP_FILES) # Threads for I/O bound tasks
NUM_PROCESS_WORKERS = os.cpu_count() or 8 # Processes for CPU bound tasks

print(f"Using {NUM_DOWNLOAD_WORKERS} download workers (threads).")
print(f"Using {NUM_PROCESS_WORKERS} PDF processing workers (processes).")

stop_all_processing = False
pdf_processing_futures = []

with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as download_executor, \
     ProcessPoolExecutor(max_workers=NUM_PROCESS_WORKERS) as process_executor:

    # Submit all download tasks
    download_futures = {download_executor.submit(download_and_extract_zip_task, zip_idx): zip_idx for zip_idx in zip_file_indices}

    for download_future in as_completed(download_futures):
        if stop_all_processing:
            break # Exit the loop if target count reached

        zip_idx = download_futures[download_future]
        pbar_zips.update(1) # Mark this ZIP's download/extraction as completed

        try:
            # Get the list of (filename, pdf_bytes) for PDFs from the downloaded ZIP
            pdf_data_list_from_zip = download_future.result() 
            
            for pdf_name_in_zip, pdf_bytes in pdf_data_list_from_zip:
                if valid_pdfs_found_count >= TARGET_PDF_COUNT:
                    stop_all_processing = True
                    break # Stop submitting new processing tasks

                # Filter out already found PDFs before submitting to processing pool
                if pdf_name_in_zip in already_found_pdf_names:
                    continue

                # Submit PDF processing to the ProcessPoolExecutor
                # pdf_bytes is copied/pickled to the new process
                future = process_executor.submit(process_single_pdf_task, pdf_name_in_zip, pdf_bytes)
                pdf_processing_futures.append(future)

        except Exception as exc:
            # print(f'ZIP {zip_idx} generated an exception during download or extraction: {exc}')
            pass # Silently pass on ZIP errors, as the task function already handles retries/logging

        # Periodically check and collect results from PDF processing futures
        # This prevents the pdf_processing_futures list from growing indefinitely large
        # and allows the main thread to update progress and the collected_data_for_json
        # list incrementally.
        completed_processing_futures = [f for f in pdf_processing_futures if f.done()]
        for completed_future in completed_processing_futures:
            pdf_processing_futures.remove(completed_future)
            try:
                result = completed_future.result()
                if result and result["filename"] not in already_found_pdf_names:
                    collected_data_for_json.append(result)
                    already_found_pdf_names.add(result["filename"])
                    valid_pdfs_found_count += 1
                    pbar_valid_pdfs.update(1)
            except Exception:
                # print(f"A PDF processing task generated an exception: {exc}")
                pass # Silently ignore errors from individual PDF processing tasks

        if valid_pdfs_found_count >= TARGET_PDF_COUNT:
            stop_all_processing = True
            print(f"\nTarget PDF count ({TARGET_PDF_COUNT}) reached. Shutting down executors and stopping new tasks.")
            break # Exit the main download loop if target reached

    # If the target is met, cancel any remaining pending tasks in both executors
    if stop_all_processing:
        for future in download_futures:
            future.cancel()
        for future in pdf_processing_futures: # Also cancel any remaining processing futures
            future.cancel()

# After all downloads are submitted/finished/cancelled,
# wait for any *remaining* PDF processing futures that were already submitted
# to complete and collect their results.
print("\nCollecting results from remaining PDF processing tasks...")
for future in as_completed(pdf_processing_futures):
    if valid_pdfs_found_count >= TARGET_PDF_COUNT:
        break # Stop if target reached during final collection
    try:
        result = future.result()
        if result and result["filename"] not in already_found_pdf_names:
            collected_data_for_json.append(result)
            already_found_pdf_names.add(result["filename"])
            valid_pdfs_found_count += 1
            pbar_valid_pdfs.update(1)
    except Exception:
        pass # Silently ignore errors from individual PDF processing tasks


pbar_valid_pdfs.close()
pbar_zips.close()

# --- Save the collected data to JSON ---
if collected_data_for_json:
    print(f"\nSaving collected text data for {len(collected_data_for_json)} PDFs to {db_path}...")
    try:
        # Sort by filename before saving for consistent output
        collected_data_for_json.sort(key=lambda x: x['filename'])
        with open(db_path, 'w', encoding='utf-8') as f_json:
            json.dump(collected_data_for_json, f_json, ensure_ascii=False, indent=2)
        print("JSON data saved successfully.")
    except IOError as e_json_io:
        print(f"Error saving JSON data: {e_json_io}")
else:
    print("\nNo valid PDFs were collected or previously existed to save to JSON.")


print(f"\n--- Processing Complete ---")
print(f"Collected {valid_pdfs_found_count} PDFs meeting the criteria ({MIN_TOKENS}-{MAX_TOKENS} tokens).")
print(f"JSON data saved in: {os.path.abspath(OUTPUT_DIR)}")

if valid_pdfs_found_count < TARGET_PDF_COUNT:
    print(f"Warning: Only found {valid_pdfs_found_count} PDFs. Targeted {TARGET_PDF_COUNT}.")


# --- Cleanup temporary ZIP download directory ---
try:
    if os.path.exists(TEMP_ZIP_DOWNLOAD_DIR):
        shutil.rmtree(TEMP_ZIP_DOWNLOAD_DIR)
        print(f"Removed temporary ZIP directory: {TEMP_ZIP_DOWNLOAD_DIR}")
except Exception as e_cleanup:
    print(f"Error during final cleanup of {TEMP_ZIP_DOWNLOAD_DIR}: {e_cleanup}")