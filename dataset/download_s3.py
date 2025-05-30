import os
import requests
import zipfile
import io
import fitz
import tiktoken
from tqdm import tqdm
import random
import time
import shutil
import json
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Ensure reproducibility for langdetect if needed
DetectorFactory.seed = 0

# --- Configuration ---
BASE_URL = "https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/"
TARGET_PDF_COUNT = 50000
MIN_TOKENS = 1000
MAX_TOKENS = 80_000

OUTPUT_DIR = "data"
# --- CHANGE THIS LINE ---
TEMP_ZIP_DOWNLOAD_DIR = "/dev/shm/temp_zip_processing" # Changed to use /dev/shm
# --- END CHANGE ---

JSON_OUTPUT_FILENAME = "collected_pdf_texts.json"

TOTAL_ZIP_FILES = 7933

db_path = os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME)

# Concurrency settings
MAX_CONCURRENT_DOWNLOADS = 50 # Max simultaneous ZIP downloads
MAX_CONCURRENT_PROCESSORS = os.cpu_count() * 2 or 8 # Max simultaneous ZIP processing (can be higher than CPU count due to I/O)

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Ensure the /dev/shm directory for temporary zips is created
os.makedirs(TEMP_ZIP_DOWNLOAD_DIR, exist_ok=True)


# Initialize tiktoken tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("Falling back to p50k_base tokenizer for tiktoken.")
    tokenizer = tiktoken.get_encoding("p50k_base")

# Global shared data and lock
# The lock protects access to collected_data_for_json, already_found_pdf_names, and valid_pdfs_found_count
data_lock = threading.Lock()
collected_data_for_json = []
already_found_pdf_names = set()
valid_pdfs_found_count = 0

# Load existing data and populate already_found_pdf_names set
if os.path.exists(db_path):
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            collected_data_for_json.extend(existing_data)
            for item in existing_data:
                already_found_pdf_names.add(item["filename"])
        print(f"Loaded {len(existing_data)} existing PDF entries from {db_path}.")
        valid_pdfs_found_count = len(already_found_pdf_names)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Could not load existing JSON database {db_path}: {e}. Starting fresh.")
else:
    print(f"No existing JSON database found at {db_path}. Starting fresh.")

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

def download_zip(zip_idx):
    """
    Downloads a single ZIP file.
    Returns local_zip_filepath on success, None on failure.
    """
    zip_url, local_zip_filepath = get_zip_url_and_local_path(zip_idx)
    try:
        # print(f"Downloading {zip_idx:04d}.zip...")
        response = requests.get(zip_url, stream=True, timeout=30) # Added timeout
        response.raise_for_status()

        with open(local_zip_filepath, 'wb') as f_zip:
            for chunk in response.iter_content(chunk_size=8192*4): # Increased chunk size for faster download
                f_zip.write(chunk)
        return local_zip_filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {zip_url}: {e}")
        if os.path.exists(local_zip_filepath): # Clean up partial download
            os.remove(local_zip_filepath)
        return None
    except Exception as e:
        print(f"Unexpected error during download of {zip_url}: {e}")
        if os.path.exists(local_zip_filepath):
            os.remove(local_zip_filepath)
        return None

def extract_text_and_count_tokens_from_pdf_data(pdf_data):
    """
    Extracts text from PDF byte data, counts tokens, and returns text and count.
    Returns (None, 0) on error during PDF processing.
    """
    full_text = ""
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        doc.close()
    except Exception as e:
        # print(f"Fitz error processing a PDF: {e}") # Optional: for debugging
        return None, 0
    if not full_text.strip():
        return "", 0
    tokens = tokenizer.encode(full_text)
    return full_text, len(tokens)

def detect_language(text):
    """
    Detects the language of the given text.
    Returns language code (e.g., 'en', 'es') or 'unknown' if detection fails.
    """
    try:
        if len(text) > 50:
            return detect(text)
        else:
            return "too_short_for_detection"
    except Exception as e:
        return "unknown"

def process_zip_file(local_zip_filepath):
    """
    Processes a downloaded ZIP file, extracts PDFs, and collects valid data.
    """
    global valid_pdfs_found_count, collected_data_for_json, already_found_pdf_names # Declare globals
    zip_idx = int(os.path.basename(local_zip_filepath).split('.')[0]) # Extract zip_idx from filename
    try:
        # print(f"Processing PDFs in {os.path.basename(local_zip_filepath)}...")
        with zipfile.ZipFile(local_zip_filepath, 'r') as zf:
            pdf_filenames_in_zip = [
                member.filename for member in zf.infolist()
                if not member.is_dir() and member.filename.lower().endswith('.pdf')
            ]

            pdfs_processed_in_zip = 0
            for pdf_name_in_zip in pdf_filenames_in_zip:
                # Check if target count is met globally before processing this PDF
                with data_lock:
                    if valid_pdfs_found_count >= TARGET_PDF_COUNT:
                        # Signal to stop processing this zip early if target hit
                        return "TARGET_REACHED"

                if pdf_name_in_zip in already_found_pdf_names:
                    continue

                try:
                    pdf_bytes = zf.read(pdf_name_in_zip)
                except Exception as e_read:
                    # print(f"Error reading {pdf_name_in_zip} from ZIP {zip_idx}: {e_read}")
                    continue

                extracted_text, token_count = extract_text_and_count_tokens_from_pdf_data(pdf_bytes)

                if extracted_text is None:
                    continue

                if MIN_TOKENS <= token_count <= MAX_TOKENS:
                    language = detect_language(extracted_text)

                    with data_lock: # Protect shared data writes
                        if valid_pdfs_found_count < TARGET_PDF_COUNT: # Double-check under lock
                            collected_data_for_json.append({
                                "filename": pdf_name_in_zip,
                                "text": extracted_text,
                                "token_count": token_count,
                                "language": language
                            })
                            already_found_pdf_names.add(pdf_name_in_zip)
                            valid_pdfs_found_count += 1
                            pdfs_processed_in_zip += 1
                            pbar_valid_pdfs.update(1)
                            # print(f"VALID: {pdf_name_in_zip}, Tokens: {token_count}, Lang: {language}. ({valid_pdfs_found_count}/{TARGET_PDF_COUNT})")
                            # If target reached, signal it
                            if valid_pdfs_found_count >= TARGET_PDF_COUNT:
                                return "TARGET_REACHED"
        return "SUCCESS" # Indicate successful processing of this ZIP
    except Exception as e:
        print(f"Error processing ZIP {zip_idx}: {e}")
        return "FAILURE"
    finally:
        # Always delete the temporary ZIP file
        if os.path.exists(local_zip_filepath):
            try:
                os.remove(local_zip_filepath)
                # print(f"Deleted temporary ZIP: {local_zip_filepath}")
            except OSError as e_del:
                print(f"Error deleting temporary ZIP {local_zip_filepath}: {e_del}")
        pbar_zips.update(1)


# --- Main Download and Processing Logic ---
zip_file_indices = list(range(TOTAL_ZIP_FILES))
# random.shuffle(zip_file_indices) # Uncomment to process ZIPs in random order

pbar_valid_pdfs = tqdm(initial=valid_pdfs_found_count, total=TARGET_PDF_COUNT, desc="Valid PDFs Found", unit="PDF")
pbar_zips = tqdm(total=len(zip_file_indices), desc="Processing ZIPs", unit="ZIP")

stop_processing_flag = False

# Use ThreadPoolExecutors for concurrent operations
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS, thread_name_prefix="Download") as download_executor, \
     ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSORS, thread_name_prefix="Process") as process_executor:

    download_futures = set()
    process_futures = set()

    # Initial seeding of download tasks
    for i in range(min(MAX_CONCURRENT_DOWNLOADS, len(zip_file_indices))):
        if valid_pdfs_found_count >= TARGET_PDF_COUNT:
            stop_processing_flag = True
            break
        zip_idx = zip_file_indices.pop(0) # Take next available zip
        future = download_executor.submit(download_zip, zip_idx)
        download_futures.add(future)
        pbar_zips.set_postfix_str(f"Downloads in flight: {len(download_futures)}")

    while download_futures or process_futures:
        # Check if target is met
        with data_lock:
            if valid_pdfs_found_count >= TARGET_PDF_COUNT:
                stop_processing_flag = True
                break

        # Process completed download tasks
        for future in as_completed(download_futures, timeout=1): # Use a timeout to regularly check other futures
            download_futures.discard(future)
            local_zip_filepath = future.result()
            if local_zip_filepath:
                process_future = process_executor.submit(process_zip_file, local_zip_filepath)
                process_futures.add(process_future)
            pbar_zips.set_postfix_str(f"Downloads in flight: {len(download_futures)}")

            # If there are more ZIPs to download and we're under the limit, submit another download
            if not stop_processing_flag and zip_file_indices and len(download_futures) < MAX_CONCURRENT_DOWNLOADS:
                zip_idx = zip_file_indices.pop(0)
                new_future = download_executor.submit(download_zip, zip_idx)
                download_futures.add(new_future)
                pbar_zips.set_postfix_str(f"Downloads in flight: {len(download_futures)}")

        # Process completed processing tasks
        for future in as_completed(process_futures, timeout=1):
            process_futures.discard(future)
            result = future.result()
            if result == "TARGET_REACHED":
                stop_processing_flag = True
                break # Break from this loop to check outer condition

        if stop_processing_flag:
            break

        # If no futures are immediately ready, wait a bit
        if not download_futures and not process_futures and zip_file_indices:
             # This means we might have exhausted initial downloads, but more are available.
             # This block ensures we submit new downloads if processing has caught up.
             # (Though the logic above for adding new downloads should handle most cases)
             if len(download_futures) < MAX_CONCURRENT_DOWNLOADS:
                if zip_file_indices:
                    zip_idx = zip_file_indices.pop(0)
                    future = download_executor.submit(download_zip, zip_idx)
                    download_futures.add(future)
                    pbar_zips.set_postfix_str(f"Downloads in flight: {len(download_futures)}")
                else:
                    break # No more zips to download
             else:
                time.sleep(0.1) # Prevent busy-waiting if nothing is ready
        elif not download_futures and not process_futures:
             # All tasks finished or no more to start
             break
        elif len(download_futures) == 0 and len(zip_file_indices) > 0 and len(process_futures) == 0:
            # If all downloads finished, but there are still zips in queue and processing is done, kick off more downloads
            for i in range(min(MAX_CONCURRENT_DOWNLOADS, len(zip_file_indices))):
                if valid_pdfs_found_count >= TARGET_PDF_COUNT:
                    stop_processing_flag = True
                    break
                zip_idx = zip_file_indices.pop(0)
                future = download_executor.submit(download_zip, zip_idx)
                download_futures.add(future)
            if stop_processing_flag:
                break


# Ensure all remaining tasks complete if target wasn't hit or during graceful shutdown
# This part might still run even if stop_processing_flag is set, but it will quickly drain.
for future in as_completed(download_futures):
    local_zip_filepath = future.result()
    if local_zip_filepath:
        process_future = process_executor.submit(process_zip_file, local_zip_filepath)
        process_futures.add(process_future)

for future in as_completed(process_futures):
    future.result() # Just to ensure exceptions are raised if any happened

pbar_valid_pdfs.close()
pbar_zips.close()


# --- Save the collected data to JSON ---
if collected_data_for_json:
    print(f"\nSaving collected text data for {len(collected_data_for_json)} PDFs to {db_path}...")
    try:
        # Sort by filename before saving for consistent output
        # Ensure we're sorting the final list after all processing is done
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

if valid_pdfs_found_count < TARGET_PDF_COUNT and not stop_processing_flag:
    print(f"Warning: Only found {valid_pdfs_found_count} PDFs after checking all available ZIPs, but targeted {TARGET_PDF_COUNT}.")
elif valid_pdfs_found_count < TARGET_PDF_COUNT and stop_processing_flag:
    print(f"Processing stopped after finding {valid_pdfs_found_count} PDFs (target was {TARGET_PDF_COUNT}).")


# --- Cleanup temporary ZIP download directory ---
try:
    if os.path.exists(TEMP_ZIP_DOWNLOAD_DIR) and not os.listdir(TEMP_ZIP_DOWNLOAD_DIR):
        shutil.rmtree(TEMP_ZIP_DOWNLOAD_DIR)
        print(f"Removed empty temporary ZIP directory: {TEMP_ZIP_DOWNLOAD_DIR}")
    elif os.path.exists(TEMP_ZIP_DOWNLOAD_DIR):
        print(f"Temporary ZIP directory {TEMP_ZIP_DOWNLOAD_DIR} is not empty. Manual check may be needed.")
except Exception as e_cleanup:
    print(f"Error during final cleanup of {TEMP_ZIP_DOWNLOAD_DIR}: {e_cleanup}")