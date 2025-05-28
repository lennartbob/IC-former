from transformers import LlamaTokenizer
from data_utils import PwCForTest 
from tqdm import tqdm
import matplotlib.pyplot as plt

def count_tokens(text, tokenizer):
    tokens = tokenizer(text)['input_ids']
    return len(tokens)

lm_path = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(lm_path, use_fast=True)

file = "data/PwC_test.jsonl" # Make sure this path is correct

dataset = PwCForTest(file)

total_tokens = 0
num_contexts = 0
context_lengths = [] # List to store token counts for plotting

print("Processing contexts and counting tokens...")
for row in tqdm(dataset):
    context, prompt, answer = row
    context_token_count = count_tokens(context, tokenizer)
    total_tokens += context_token_count
    num_contexts += 1
    context_lengths.append(context_token_count) # Store each context's token count

# Calculate the average
average_token_length = total_tokens / num_contexts if num_contexts > 0 else 0

print(f"\nTotal contexts processed: {num_contexts}")
print(f"Total tokens in contexts: {total_tokens}")
print(f"Average token length of the context: {average_token_length:.2f}")

# --- Plotting the Distribution ---
if context_lengths:
    plt.figure(figsize=(12, 7))
    plt.hist(context_lengths, bins=50, color='#6A5ACD', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Context Lengths (in Tokens)', fontsize=16)
    plt.xlabel('Number of Tokens', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("No context lengths to plot. Please ensure the dataset is populated.")

