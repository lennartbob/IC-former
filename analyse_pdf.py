import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

def analyze_json_data(json_file_path):
    """
    Loads data from a JSON file, prints file size, shows first 2 items,
    and plots the distribution of token lengths.
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

    # 3. Show the first 2 items (or fewer if less than 2 items exist)
    print(f"\n--- First {min(2, len(data))} Item(s) from the JSON data ---")
    for i, item in enumerate(data[:2]):
        print(f"\nItem {i + 1}:")
        # To avoid printing very long text, we can show a snippet or just keys
        item_to_show = {
            "filename": item.get("filename", "N/A"),
            "token_count": item.get("token_count", "N/A"),
            "text_snippet": (item.get("text", "")[:100] + "...") if item.get("text") else "N/A"
        }
        pprint.pprint(item_to_show, indent=2, width=120) # Using pprint for readability

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
        print("\nNo valid 'token_count' data found in the JSON to plot.")
        return

    # 5. Plot the distribution of token lengths
    print(f"\n--- Plotting Token Length Distribution ({len(token_lengths)} documents) ---")
    plt.figure(figsize=(12, 7)) # Increased height for better title/label spacing

    # Using seaborn for a slightly nicer histogram with KDE
    sns.histplot(token_lengths, kde=True, bins=50, color='skyblue', edgecolor='black')

    mean_tokens = sum(token_lengths) / len(token_lengths)
    median_tokens = sorted(token_lengths)[len(token_lengths) // 2]

    plt.axvline(mean_tokens, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_tokens:.0f}')
    plt.axvline(median_tokens, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_tokens:.0f}')

    plt.title('Distribution of Token Lengths in Collected PDFs', fontsize=16)
    plt.xlabel('Token Count per PDF', fontsize=14)
    plt.ylabel('Frequency (Number of PDFs)', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjusts plot to prevent labels from overlapping
    
    plot_filename = os.path.join(os.path.dirname(json_file_path) or '.', "token_length_distribution.png")
    try:
        plt.savefig(plot_filename)
        print(f"Distribution plot saved as: {plot_filename}")
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    plt.show()
    print("Plot display window closed.")


if __name__ == "__main__":
    # Construct the path to the JSON file
    # Assumes this script is run from the same directory level as the previous script,
    # or that DEFAULT_OUTPUT_DIR is accessible.
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level if DEFAULT_OUTPUT_DIR is a sibling to the script's parent dir
    # Or adjust as per your directory structure.
    # For simplicity, assuming DEFAULT_OUTPUT_DIR is relative to where you run the script:
    json_path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_JSON_FILENAME)

    # You can override the path if needed:
    # json_path = input(f"Enter path to JSON file (default: {json_path}): ") or json_path
    
    analyze_json_data(json_path)