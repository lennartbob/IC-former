import json
import os
import random

def split_jsonl_dataset(input_file_path, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Splits a JSONL file into training, testing, and validation sets.

    Args:
        input_file_path (str): The path to the input JSONL file.
        train_ratio (float): The proportion of data for the training set (default: 0.8).
        test_ratio (float): The proportion of data for the testing set (default: 0.1).
        val_ratio (float): The proportion of data for the validation set (default: 0.1).
    """

    # Ensure ratios sum up to 1 (or close to it due to floating point precision)
    total_ratio = train_ratio + test_ratio + val_ratio
    if not (0.99 <= total_ratio <= 1.01):
        print(f"Warning: Ratios do not sum up to 1.0. Current sum: {total_ratio}")
        print("Adjusting ratios proportionally to sum to 1.0.")
        train_ratio /= total_ratio
        test_ratio /= total_ratio
        val_ratio /= total_ratio

    data = []
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{input_file_path}': {e}")
        print("Please ensure each line is a valid JSON object.")
        return

    random.shuffle(data) # Shuffle the data to ensure random distribution

    num_samples = len(data)
    num_train = int(num_samples * train_ratio)
    num_test = int(num_samples * test_ratio)
    # The rest goes to validation to account for potential rounding errors
    num_val = num_samples - num_train - num_test

    train_data = data[:num_train]
    test_data = data[num_train : num_train + num_test]
    val_data = data[num_train + num_test : num_train + num_test + num_val]

    # Determine the base name and extension for output files
    base_name, ext = os.path.splitext(input_file_path)
    output_dir = os.path.dirname(input_file_path)
    if not output_dir: # If no directory specified, use current directory
        output_dir = '.'
    file_name_without_path = os.path.basename(base_name)

    # Define output file paths
    train_output_path = os.path.join(output_dir, f"train_{file_name_without_path}{ext}")
    test_output_path = os.path.join(output_dir, f"test_{file_name_without_path}{ext}")
    val_output_path = os.path.join(output_dir, f"val_{file_name_without_path}{ext}")

    print(f"Total samples: {num_samples}")
    print(f"Train samples: {len(train_data)} ({len(train_data)/num_samples:.2%})")
    print(f"Test samples: {len(test_data)} ({len(test_data)/num_samples:.2%})")
    print(f"Validation samples: {len(val_data)} ({len(val_data)/num_samples:.2%})")

    # Write data to respective files
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"Train data saved to: {train_output_path}")

    with open(test_output_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    print(f"Test data saved to: {test_output_path}")

    with open(val_output_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    print(f"Validation data saved to: {val_output_path}")

# --- How to use the script ---
if __name__ == "__main__":
    # Define the path to your input JSONL file
    # Assuming your file is named SFT_con_sum.jsonl inside the 'data' directory
    input_jsonl_file = "data/SFT_con_sum.jsonl"

    # Call the function to split the dataset
    split_jsonl_dataset(input_jsonl_file, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1)

