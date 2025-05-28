import datasets
import tiktoken
import random
from tqdm.auto import tqdm
import re
from collections import Counter
import unicodedata # For unicode category checks

# --- Main Dataset Creation Function ---
def create_filtered_dataset(
    source_dataset_name="machelreid/m2d2",
    source_config_name="as",
    output_dataset_path="./m2d2_filtered_long_clean_texts", # New output path
    min_tokens=3000,
    max_texts_to_sample=2000,
):

    try:
        # Load the dataset
        dataset = datasets.load_dataset(source_dataset_name, source_config_name, trust_remote_code=True)
        train_dataset = dataset['train']

        # Initialize the tiktoken encoder
        enc = tiktoken.get_encoding("cl100k_base")

        candidate_texts = [] # For texts that pass pollution check and token count (3k-15k range)
        high_token_texts = [] # For texts > 15k that pass pollution check

        print(f"Processing {len(train_dataset)} examples from the 'train' split...")

        # Filter texts based on pollution and token count
        for i, example in enumerate(tqdm(train_dataset, desc="Filtering and checking pollution")):
            if 'text' in example and example['text'] is not None:
                text = example['text']

                # If not polluted, then count tokens
                tokens = enc.encode(text, allowed_special="all") # Allow special tokens
                token_count = len(tokens)

                if token_count >= min_tokens:
                    candidate_texts.append({'text': text})
        # Randomly sample from the 'candidate_texts'
        num_to_sample = min(max_texts_to_sample, len(candidate_texts))
        sampled_texts = random.sample(candidate_texts, num_to_sample)
        print(f"Randomly sampled {len(sampled_texts)} texts from the primary range.")

        # Combine sampled texts with all high_token_texts
        final_selected_texts = sampled_texts + high_token_texts
        print(f"Total clean texts selected for the new dataset: {len(final_selected_texts)}")

        if not final_selected_texts:
            print("No texts matched the criteria and pollution checks. New dataset will be empty.")
            return

        # Create a new Dataset object
        new_dataset = datasets.Dataset.from_list(final_selected_texts)

        # Save the new dataset
        new_dataset.save_to_disk(output_dataset_path)
        print(f"New dataset saved to: '{output_dataset_path}', num of texts: {len(final_selected_texts)}")
        print("\nTo load this dataset later, use:")
        print(f"   new_data = datasets.load_from_disk('{output_dataset_path}')")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("Please ensure:")
        print("1. You have `datasets`, `tiktoken`, `matplotlib`, and `tqdm` installed (`pip install datasets tiktoken matplotlib tqdm`).")
        print("2. You have an active internet connection to download the source dataset.")
        print(f"3. The configuration name '{source_config_name}' is valid for '{source_dataset_name}'.")

# --- Run the function ---
if __name__ == "__main__":
    create_filtered_dataset(
        min_tokens=1000,
        max_texts_to_sample=2000,
    )