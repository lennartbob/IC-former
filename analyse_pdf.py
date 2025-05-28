import json
import os
import io # Import io for string output
from collections import Counter
import pprint # For pretty printing complex data structures

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "downloaded_valid_pdfs"
DEFAULT_JSON_FILENAME = "/mnt/data_volume/dataset/downloaded_valid_pdfs/collected_pdf_texts.json"

def get_human_readable_size(size_bytes):
    """Converts a size in bytes to a human-readable format (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_name[i]}"

def analyze_json_data_for_terminal(json_file_path):
    """
    Loads data from a JSON file, prints file size, shows first 2 items with full text,
    and prints a text-based distribution of token lengths.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at '{json_file_path}'")
        return

    # 1. Get and print file size
    try:
        file_size_bytes = os.path.getsize(json_file_path)
        readable_size = get_human_readable_size(file_size_bytes)
        print(f"--- JSON File Information ---")
        print(f"File: {json_file_path}")
        print(f"Size: {file_size_bytes} bytes ({readable_size})")
    except OSError as e:
        print(f"Error getting file size for {json_file_path}: {e}")
        return

    # 2. Load JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. The file might be corrupted or not valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_file_path}: {e}")
        return

    if not isinstance(data, list) or not data:
        print("JSON data is empty or not in the expected list format.")
        return

    # 3. Show the first 2 items (or fewer if less than 2 items exist) with full text
    print(f"\n--- First {min(2, len(data))} Item(s) from the JSON data (Full Text) ---")
    for i, item in enumerate(data[:2]):
        print(f"\nItem {i + 1}:")
        item_to_show = {
            "filename": item.get("filename", "N/A"),
            "token_count": item.get("token_count", "N/A"),
            "text": item.get("text", "N/A") # Show full text
        }
        pprint.pprint(item_to_show, indent=2, width=120)

    # 4. Extract token lengths
    token_lengths = []
    for item in data:
        if isinstance(item, dict) and "token_count" in item:
            try:
                token_lengths.append(int(item["token_count"]))
            except (ValueError, TypeError):
                print(f"Warning: Invalid token_count found for item (filename: {item.get('filename', 'Unknown')}): {item['token_count']}")
        else:
            filename_info = item.get('filename', 'an unknown item') if isinstance(item, dict) else 'an item'
            print(f"Warning: Item missing 'token_count' key or not a dictionary: {filename_info}")

    if not token_lengths:
        print("\nNo valid 'token_count' data found in the JSON to analyze.")
        return

    # 5. Print the distribution of token lengths as a string
    print(f"\n--- Token Length Distribution Statistics ({len(token_lengths)} documents) ---")

    min_tokens = min(token_lengths)
    max_tokens = max(token_lengths)
    mean_tokens = sum(token_lengths) / len(token_lengths)
    median_tokens = sorted(token_lengths)[len(token_lengths) // 2]

    # Calculate quartiles
    q1_index = (len(token_lengths) - 1) // 4
    q3_index = (len(token_lengths) - 1) * 3 // 4
    q1_tokens = sorted(token_lengths)[q1_index]
    q3_tokens = sorted(token_lengths)[q3_index]

    print(f"  Minimum Token Count: {min_tokens}")
    print(f"  Maximum Token Count: {max_tokens}")
    print(f"  Mean Token Count: {mean_tokens:.2f}")
    print(f"  Median Token Count: {median_tokens}")
    print(f"  25th Percentile (Q1): {q1_tokens}")
    print(f"  75th Percentile (Q3): {q3_tokens}")

    # For a more visual but text-based distribution, we can use a simple histogram
    # by grouping into bins.
    print("\n--- Text-Based Histogram of Token Lengths (Approximate) ---")
    
    # Determine appropriate bin size based on range
    token_range = max_tokens - min_tokens
    num_bins = 20 # You can adjust this for more or fewer bars
    bin_width = token_range / num_bins
    
    if bin_width == 0: # Handle cases where all token counts are the same
        print(f"All documents have approximately {min_tokens} tokens.")
        return

    bin_counts = Counter()
    for length in token_lengths:
        bin_index = min(int((length - min_tokens) / bin_width), num_bins - 1)
        bin_counts[bin_index] += 1
    
    max_bin_count = max(bin_counts.values()) if bin_counts else 0
    
    if max_bin_count == 0:
        print("No data to display in histogram.")
        return

    bar_char = '#'
    scale_factor = 50.0 / max_bin_count # Scale bars to fit terminal width

    for i in range(num_bins):
        lower_bound = min_tokens + i * bin_width
        upper_bound = min_tokens + (i + 1) * bin_width
        count = bin_counts.get(i, 0)
        bar = bar_char * int(count * scale_factor)
        print(f"[{lower_bound:6.0f} - {upper_bound:6.0f}): {count:5d} |{bar}")


if __name__ == "__main__":
    # Construct the path to the JSON file
    # This assumes DEFAULT_OUTPUT_DIR is a direct child of the directory
    # where this script is executed.
    
    # If DEFAULT_JSON_FILENAME includes a full path like '/mnt/data_volume/dataset/downloaded_valid_pdfs/collected_pdf_texts.json',
    # then `os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_JSON_FILENAME)` won't work as intended
    # because os.path.join ignores preceding path components if a subsequent component
    # is an absolute path.
    # So, we should just use DEFAULT_JSON_FILENAME directly if it's already an absolute path.

    json_path = DEFAULT_JSON_FILENAME # Use the provided absolute path

    # If your JSON file is truly relative to DEFAULT_OUTPUT_DIR,
    # you might need to adjust this. But given the example, it looks absolute.
    # For example, if DEFAULT_JSON_FILENAME was "collected_pdf_texts.json"
    # and you wanted it in "downloaded_valid_pdfs" relative to the script:
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # json_path = os.path.join(current_script_dir, DEFAULT_OUTPUT_DIR, "collected_pdf_texts.json")

    analyze_json_data_for_terminal(json_path)