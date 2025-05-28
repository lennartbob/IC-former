import os
import requests
import zipfile
import io
import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm
import random
import time
import shutil # For moving files

# --- Configuration ---
BASE_URL = "https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/"
TARGET_PDF_COUNT = 2000 # How many valid PDFs to collect
MIN_TOKENS = 1500
OUTPUT_DIR = "downloaded_valid_pdfs"
TEMP_DIR = "temp_pdf_processing" # For temporary extraction

# Total number of unique PDFs in the corpus (0000000.pdf to 7932877.pdf)
# From README: "All PDF files are named using a sequential 7-digit number ... through 7932877.pdf"
TOTAL_UNIQUE_PDFS = 7932878 # (0 to 7932877 inclusive)
MAX_PDF_NUM = TOTAL_UNIQUE_PDFS - 1

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize tiktoken tokenizer (GPT-2 is common, or choose another)
# Using cl100k_base as it's common for newer models like GPT-3.5/4
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    print("Falling back to p50k_base tokenizer")
    tokenizer = tiktoken.get_encoding("p50k_base")


def count_tokens_from_pdf_data(pdf_data):
    """Extracts text from PDF data and counts tokens."""
    text = ""
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        # print(f"Fitz error processing PDF: {e}") # Comment out for less verbose output
        return 0 # Can't process, so 0 tokens

    if not text.strip():
        return 0

    tokens = tokenizer.encode(text)
    return len(tokens)

def get_pdf_details(pdf_sequential_number):
    """
    Generates the ZIP filename, subdirectory, and PDF filename within the ZIP
    based on the global sequential PDF number.
    """
    pdf_filename_in_zip = f"{pdf_sequential_number:07d}.pdf"

    # Each ZIP contains up to 1,000 PDF files
    zip_num_int = pdf_sequential_number // 1000
    zip_filename = f"{zip_num_int:04d}.zip"

    # ZIP files are clustered into groups of 1,000
    # e.g., 0000.zip to 0999.zip are in zipfiles/0000-0999/
    # 1000.zip to 1999.zip are in zipfiles/1000-1999/
    zip_group_start = (zip_num_int // 1000) * 1000
    zip_group_end = zip_group_start + 999
    zip_subdir = f"zipfiles/{zip_group_start:04d}-{zip_group_end:04d}/"

    zip_url = f"{BASE_URL}{zip_subdir}{zip_filename}"
    return zip_url, pdf_filename_in_zip, zip_filename


# --- Main Download and Processing Logic ---
collected_pdfs_info = []
processed_pdf_numbers = set() # To avoid reprocessing the same random PDF number if selected again

# We will iterate more than TARGET_PDF_COUNT times because many PDFs won't meet the criteria
# The tqdm total can be an estimate or updated dynamically if we knew the acceptance rate.
# For simplicity, let's set a large number for attempts or just let it run.
# The progress bar will show how many valid PDFs we *found*.
pbar = tqdm(total=TARGET_PDF_COUNT, desc="Finding valid PDFs")

# Keep track of currently downloaded ZIP to avoid re-downloading if multiple
# PDFs are selected from the same ZIP in close succession (less likely with pure random)
# A more advanced cache could store multiple ZIPs or their contents.
current_zip_path = None
current_zip_url_loaded = None

attempts = 0
max_attempts = TOTAL_UNIQUE_PDFS * 2 # Heuristic limit to prevent infinite loops if criteria are too strict

while len(collected_pdfs_info) < TARGET_PDF_COUNT and attempts < max_attempts:
    attempts += 1
    
    # 1. Randomly select a PDF number
    pdf_sequential_number = random.randint(0, MAX_PDF_NUM)

    if pdf_sequential_number in processed_pdf_numbers:
        continue # Skip if we already tried this number
    processed_pdf_numbers.add(pdf_sequential_number)

    # 2. Get URL details for this PDF
    zip_url, pdf_filename_in_zip, zip_filename_short = get_pdf_details(pdf_sequential_number)
    
    temp_zip_path = os.path.join(TEMP_DIR, zip_filename_short)
    extracted_pdf_path = os.path.join(TEMP_DIR, pdf_filename_in_zip)

    try:
        # 3. Download the ZIP file (if not already the current one)
        # This is a simple cache for only the very last ZIP.
        # A more robust solution would be to download a ZIP, extract all PDFs, process them,
        # then move to the next ZIP. Current approach is truly random PDF selection.
        if current_zip_url_loaded != zip_url:
            print(f"\nDownloading {zip_url}...") # Verbose
            response = requests.get(zip_url, stream=True)
            response.raise_for_status() # Check for HTTP errors
            
            with open(temp_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            current_zip_path = temp_zip_path
            current_zip_url_loaded = zip_url
        else:
            # print(f"\nUsing cached {zip_filename_short} for {pdf_filename_in_zip}") # Verbose
            pass


        # 4. Extract the specific PDF from the ZIP
        print(f"Extracting {pdf_filename_in_zip} from {current_zip_path}...") # Verbose
        with zipfile.ZipFile(current_zip_path, 'r') as zf:
            # Check if the specific PDF is in this ZIP (it should be by naming convention)
            if pdf_filename_in_zip not in zf.namelist():
                # This might happen if the last ZIP in a group doesn't have 1000 files
                # or due to the Errata (missing files).
                # print(f"Warning: {pdf_filename_in_zip} not found in {zip_filename_short}. Skipping.")
                continue
            
            pdf_data = zf.read(pdf_filename_in_zip)

        # 5. Count tokens
        # print(f"Processing {pdf_filename_in_zip} for token count...") # Verbose
        num_tokens = count_tokens_from_pdf_data(pdf_data)

        # 6. If token count > MIN_TOKENS:
        if num_tokens > MIN_TOKENS:
            final_pdf_path = os.path.join(OUTPUT_DIR, pdf_filename_in_zip)
            with open(final_pdf_path, 'wb') as f_out:
                f_out.write(pdf_data)
            
            collected_pdfs_info.append({
                "filename": pdf_filename_in_zip,
                "zip_url": zip_url,
                "tokens": num_tokens
            })
            pbar.update(1)
            print(f"VALID: {pdf_filename_in_zip}, Tokens: {num_tokens}. Saved.")

            # If the current_zip_path (temp_zip_path) still exists and we are done with this ZIP
            # (or to save space), we could delete it here.
            # However, if the next random PDF is from the same ZIP, we'd redownload.
            # For this truly random strategy, we keep the last ZIP.

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # print(f"ZIP file not found: {zip_url}. This might be expected for last few ZIPs or errata. Skipping.")
            pass # Common if a zip doesn't exist (e.g. if MAX_PDF_NUM isn't perfectly divisible)
        else:
            # print(f"HTTP error downloading {zip_url}: {e}")
            pass
        time.sleep(1) # Wait a bit after an error
    except zipfile.BadZipFile:
        # print(f"Error: Bad ZIP file {current_zip_path}. Skipping.")
        if os.path.exists(current_zip_path): os.remove(current_zip_path) # Remove corrupted download
        current_zip_path = None # Force re-download if tried again
        current_zip_url_loaded = None
        time.sleep(1)
    except Exception as e:
        # print(f"An unexpected error occurred for {pdf_filename_in_zip} from {zip_url}: {e}")
        time.sleep(1)
    finally:
        # Clean up extracted individual PDF if it exists (not strictly necessary with in-memory)
        if os.path.exists(extracted_pdf_path):
            try:
                os.remove(extracted_pdf_path)
            except OSError:
                pass # file might be locked briefly on Windows

pbar.close()

print(f"\n--- Download and Processing Complete ---")
print(f"Collected {len(collected_pdfs_info)} PDFs meeting the criteria (>{MIN_TOKENS} tokens).")
print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")
# print(f"Total attempts to find PDFs: {attempts}")

# --- Cleanup temporary ZIP files ---
print("Cleaning up temporary files...")
try:
    shutil.rmtree(TEMP_DIR)
    print(f"Removed temporary directory: {TEMP_DIR}")
except Exception as e:
    print(f"Error removing temporary directory {TEMP_DIR}: {e}")


# --- Optional: Report on Country Distribution of the Source Corpus ---
# This part is for context, as the script doesn't actively match this distribution.
# You would need to download and parse 'cc-hosts-20230303.csv.gz'
print("\n--- Source Corpus Country Distribution (Top 10 from README) ---")
print("Note: The downloaded sample is random and may not perfectly mirror this.")
print("Country Code | Count")
print("US           | 3,259,209")
print("DE           | 896,990")
print("FR           | 462,215")
print("JP           | 364,303")
print("GB           | 268,950")
print("IT           | 228,065")
print("NL           | 206,389")
print("RU           | 176,947")
print("CA           | 175,853")
print("ES           | 173,619")
print("-----------------------------------------------------------------")