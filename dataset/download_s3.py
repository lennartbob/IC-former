import os
import requests
import zipfile
import io
import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm
import random  # For shuffling ZIP order if desired
import time
import shutil  # For directory cleanup
import json  # For JSON output
from langdetect import detect, DetectorFactory # For language detection

# Ensure reproducibility for langdetect if needed
DetectorFactory.seed = 0

# --- Configuration ---
BASE_URL = "https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/"
TARGET_PDF_COUNT = 50000  # How many valid PDFs to collect
MIN_TOKENS = 1500
MAX_TOKENS = 50_000 # Added max token limit

# OUTPUT_DIR will now only contain the JSON file, not individual PDFs
OUTPUT_DIR = "processed_pdf_data" # Renamed to reflect its new purpose
# Temporary directory for storing the currently downloaded ZIP file
TEMP_ZIP_DOWNLOAD_DIR = "temp_zip_processing"
JSON_OUTPUT_FILENAME = "collected_pdf_texts.json"

# Corpus details (from README)
# PDFs: 0000000.pdf to 7932877.pdf
# ZIPs: 0000.zip to 7932.zip (7,933 ZIP files)
TOTAL_ZIP_FILES = 7933

db_path = os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME)

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_ZIP_DOWNLOAD_DIR, exist_ok=True)

# Initialize tiktoken tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("Falling back to p50k_base tokenizer for tiktoken.")
    tokenizer = tiktoken.get_encoding("p50k_base")

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

valid_pdfs_found_count = len(already_found_pdf_names) # Initialize count with existing PDFs

def get_zip_url_and_local_path(zip_num_int):
    """
    Generates the URL for a given ZIP number and its intended local download path.
    """
    zip_filename_short = f"{zip_num_int:04d}.zip"
    # Determine subdirectory for the ZIP file
    # e.g., 0000.zip to 0999.zip are in zipfiles/0000-0999/
    # 1000.zip to 1999.zip are in zipfiles/1000-1999/
    zip_group_start = (zip_num_int // 1000) * 1000
    zip_group_end = zip_group_start + 999
    zip_subdir_path = f"zipfiles/{zip_group_start:04d}-{zip_group_end:04d}/"

    full_zip_url = f"{BASE_URL}{zip_subdir_path}{zip_filename_short}"
    local_download_path = os.path.join(TEMP_ZIP_DOWNLOAD_DIR, zip_filename_short)
    return full_zip_url, local_download_path

def extract_text_and_count_tokens_from_pdf_data(pdf_data):
    """
    Extracts text from PDF byte data, counts tokens, and returns text and count.
    Returns (None, 0) on error during PDF processing.
    """
    full_text = ""
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        doc.close()
    except Exception as e:
        # print(f"Fitz error processing a PDF: {e}") # Optional: for debugging
        return None, 0  # Indicate error

    if not full_text.strip():
        return "", 0  # No text extracted

    tokens = tokenizer.encode(full_text)
    return full_text, len(tokens)

def detect_language(text):
    """
    Detects the language of the given text.
    Returns language code (e.g., 'en', 'es') or 'unknown' if detection fails.
    """
    try:
        # Langdetect can raise a LangDetectException if it can't detect.
        # It also expects a certain amount of text.
        if len(text) > 50: # Only try to detect if enough text is present
            return detect(text)
        else:
            return "too_short_for_detection"
    except Exception as e:
        # print(f"Language detection error: {e}") # Optional: for debugging
        return "unknown"

# --- Main Download and Processing Logic ---

# Create a list of ZIP file numbers (0 to 7932)
# You can shuffle this list if you want to process ZIPs in a random order
# to get a more diverse sample early on if you don't intend to run through all.
zip_file_indices = list(range(TOTAL_ZIP_FILES))
# random.shuffle(zip_file_indices) # Uncomment to process ZIPs in random order

# Progress bar for overall valid PDFs found
pbar_valid_pdfs = tqdm(initial=valid_pdfs_found_count, total=TARGET_PDF_COUNT, desc="Valid PDFs Found", unit="PDF")
# Progress bar for ZIPs processed
pbar_zips = tqdm(total=len(zip_file_indices), desc="Processing ZIPs", unit="ZIP")

stop_all_processing = False

for zip_idx in zip_file_indices:
    if stop_all_processing:
        break

    pbar_zips.set_postfix_str(f"Current ZIP: {zip_idx:04d}.zip")
    zip_url, local_zip_filepath = get_zip_url_and_local_path(zip_idx)

    try:
        # 1. Download the current ZIP file
        # print(f"\nDownloading {zip_url}...")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(local_zip_filepath, 'wb') as f_zip:
            for chunk in response.iter_content(chunk_size=8192*2): # Increased chunk size
                f_zip.write(chunk)

        # 2. Process PDFs within the downloaded ZIP
        # print(f"Processing PDFs in {local_zip_filepath}...")
        with zipfile.ZipFile(local_zip_filepath, 'r') as zf:
            # Get list of all PDF files in the ZIP
            pdf_filenames_in_zip = [
                member.filename for member in zf.infolist()
                if not member.is_dir() and member.filename.lower().endswith('.pdf')
            ]
            # pdf_filenames_in_zip.sort() # Optional: process in order

            for pdf_name_in_zip in pdf_filenames_in_zip:
                if valid_pdfs_found_count >= TARGET_PDF_COUNT:
                    stop_all_processing = True
                    break # Stop processing PDFs in this ZIP

                # Filter out already found PDFs
                if pdf_name_in_zip in already_found_pdf_names:
                    # print(f"Skipping {pdf_name_in_zip}: Already in database.")
                    continue

                try:
                    pdf_bytes = zf.read(pdf_name_in_zip) # PDF content loaded into RAM here
                except Exception as e_read:
                    # print(f"Error reading {pdf_name_in_zip} from ZIP: {e_read}")
                    continue # Skip this PDF

                # 3. Extract text and count tokens
                extracted_text, token_count = extract_text_and_count_tokens_from_pdf_data(pdf_bytes)
                # pdf_bytes is now eligible for garbage collection

                if extracted_text is None: # fitz processing error
                    continue # Skip this PDF

                # 4. If token count meets criteria (min and max)
                if MIN_TOKENS <= token_count <= MAX_TOKENS: # Added MAX_TOKENS check
                    # Detect language
                    language = detect_language(extracted_text)

                    # --- IMPORTANT CHANGE: No longer saving the PDF file itself ---
                    # The following lines are REMOVED:
                    # output_pdf_path = os.path.join(OUTPUT_DIR, pdf_name_in_zip)
                    # with open(output_pdf_path, 'wb') as f_out_pdf:
                    #     f_out_pdf.write(pdf_bytes)

                    # Store data for JSON
                    collected_data_for_json.append({
                        "filename": pdf_name_in_zip,
                        "text": extracted_text,
                        "token_count": token_count,
                        "language": language # Added language
                    })
                    already_found_pdf_names.add(pdf_name_in_zip) # Add to set to prevent re-processing

                    valid_pdfs_found_count += 1
                    pbar_valid_pdfs.update(1)
                    # print(f"VALID: {pdf_name_in_zip}, Tokens: {token_count}, Lang: {language}. ({valid_pdfs_found_count}/{TARGET_PDF_COUNT})")

            if stop_all_processing: # Check again if target reached within inner loop
                break

    except requests.exceptions.RequestException as e_req:
        print(f"Network error downloading {zip_url}: {e_req}. Skipping this ZIP.")
        time.sleep(2) # Wait a bit before trying next ZIP
    finally:
        # 5. Delete the downloaded ZIP file to save space
        if os.path.exists(local_zip_filepath):
            try:
                os.remove(local_zip_filepath)
                # print(f"Deleted temporary ZIP: {local_zip_filepath}")
            except OSError as e_del:
                print(f"Error deleting temporary ZIP {local_zip_filepath}: {e_del}")
        pbar_zips.update(1)


pbar_valid_pdfs.close()
pbar_zips.close()

# --- Save the collected data to JSON ---
# This will overwrite the existing JSON with the updated list including new PDFs
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

if valid_pdfs_found_count < TARGET_PDF_COUNT and not stop_all_processing:
    print(f"Warning: Only found {valid_pdfs_found_count} PDFs after checking all available ZIPs, but targeted {TARGET_PDF_COUNT}.")
elif valid_pdfs_found_count < TARGET_PDF_COUNT and stop_all_processing:
    print(f"Processing stopped after finding {valid_pdfs_found_count} PDFs (target was {TARGET_PDF_COUNT}).")


# --- Cleanup temporary ZIP download directory ---
# It should be empty if all ZIPs were deleted, but we can try to remove it.
try:
    if os.path.exists(TEMP_ZIP_DOWNLOAD_DIR) and not os.listdir(TEMP_ZIP_DOWNLOAD_DIR):
        shutil.rmtree(TEMP_ZIP_DOWNLOAD_DIR)
        print(f"Removed empty temporary ZIP directory: {TEMP_ZIP_DOWNLOAD_DIR}")
    elif os.path.exists(TEMP_ZIP_DOWNLOAD_DIR):
        print(f"Temporary ZIP directory {TEMP_ZIP_DOWNLOAD_DIR} is not empty. Manual check may be needed.")
except Exception as e_cleanup:
    print(f"Error during final cleanup of {TEMP_ZIP_DOWNLOAD_DIR}: {e_cleanup}")