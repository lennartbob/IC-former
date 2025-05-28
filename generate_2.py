import os
import json
import torch
import argparse
from tqdm import tqdm
import time # Import the time module for timing
import matplotlib.pyplot as plt # Import matplotlib for plotting

from transformers import AutoModelForCausalLM, LlamaTokenizer
from data_utils import PwCForTest
from icformer import ICFormerModel
from modules import ICFormerQA
from utils import parse_args

def test_generate():
    data_path = "./data/PwC_test.jsonl"
    lm_path = "meta-llama/Llama-2-7b-hf"
    icformer_path = "./finetune2"
    save_path = "./results"
    file_name = "PwC_output.jsonl"
    max_new_tokens = 256
    
    # Limit the number of lines to process
    num_lines_to_process = 100 

    data = PwCForTest(data_path)
    tokenizer = LlamaTokenizer.from_pretrained(lm_path, use_fast=True)

    icformer = ICFormerModel.from_pretrained(icformer_path, device_map="cuda", torch_dtype=torch.bfloat16)
    icformer.requires_grad_(False)
    language_model = AutoModelForCausalLM.from_pretrained(lm_path, device_map="cuda", torch_dtype=torch.bfloat16)
    language_model.requires_grad_(False)

    model = ICFormerQA(icformer, language_model, tokenizer)

    ckpt = torch.load(os.path.join(icformer_path, 'param.pt'))
    with torch.no_grad():
        model.digest_embeddings.copy_(ckpt['digest_embeddings'])
        model.FT.copy_(ckpt['FT'])

    cache = []
    os.makedirs(save_path, exist_ok=True)

    total_inference_time = 0.0
    total_output_tokens = 0
    num_generations = 0

    # Lists to store data for plotting
    inference_times_list = []
    output_tokens_list = []

    # Adjust tqdm total to reflect the limited number of lines
    with tqdm(total=min(len(data), num_lines_to_process), ncols=100, unit='B') as pbar:
        for i in range(min(len(data), num_lines_to_process)): # Iterate only up to num_lines_to_process
            context, prompt, answer = data[i]

            context_ids = model.tokenizer(context)['input_ids']
            prompt_ids = model.tokenizer(prompt)['input_ids']

            context_embeds = model.convert_ids_to_embeds(context_ids)
            prompt_embeds = model.convert_ids_to_embeds(prompt_ids)
            soft_prompt = model.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=model.use_chunk)
            inputs_embeds = torch.cat([model.pre_embeds, soft_prompt, model.FT, prompt_embeds, model.post_embeds], dim=1)

            start_time = time.time() # Start timing
            outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, streaming=False)[0]
            end_time = time.time() # End timing

            inference_time = end_time - start_time
            
            # Determine the actual generated text and its token length based on the type of 'outputs'
            generated_text = ""
            output_token_count = 0

            if isinstance(outputs, str):
                # If outputs is already a string, use it directly
                generated_text = outputs
                # Tokenize the string to get its length in tokens for stats
                output_token_count = len(tokenizer.encode(generated_text, add_special_tokens=False))
            elif isinstance(outputs, torch.Tensor):
                # If it's a tensor, ensure it's of type long and then decode
                if outputs.dtype != torch.long:
                    outputs = outputs.to(torch.long)
                generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
                output_token_count = len(outputs) # Length of the token ID tensor
            elif isinstance(outputs, list) and all(isinstance(x, int) for x in outputs):
                # If it's a list of integers, decode it
                generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
                output_token_count = len(outputs) # Length of the list of token IDs
            else:
                # Fallback for unexpected types, attempt to convert to string and log a warning
                print(f"Warning: 'model.generate' returned an unexpected type: {type(outputs)}. Attempting to convert to string.")
                try:
                    generated_text = str(outputs)
                    output_token_count = len(tokenizer.encode(generated_text, add_special_tokens=False))
                except Exception as e:
                    print(f"Error converting unexpected 'outputs' type to string: {e}")
                    generated_text = "Error: Could not decode output due to unexpected type."
                    output_token_count = 0 # Cannot determine length

            total_inference_time += inference_time
            total_output_tokens += output_token_count
            num_generations += 1

            inference_times_list.append(inference_time)
            output_tokens_list.append(output_token_count)

            cache.append({'input':context, 'prompt':prompt, 'answer':generated_text, "output_tokens":output_token_count, "inference_time":inference_time})
            pbar.update(1)

            # Save results periodically or at the end of the limited processing
            if (i + 1) % 50 == 0 or (i + 1) == min(len(data), num_lines_to_process):
                with open(os.path.join(save_path, file_name), 'a') as f:
                    for item in cache:
                        json.dump(item, f)
                        f.write('\n')
                cache = []

    # Calculate and print statistics
    if num_generations > 0:
        average_inference_time = total_inference_time / num_generations
        average_output_length = total_output_tokens / num_generations
        tokens_per_second = total_output_tokens / total_inference_time if total_inference_time > 0 else 0

        print("\n--- Inference Statistics ---")
        print(f"Number of generations processed: {num_generations}")
        print(f"Average total inference time: {average_inference_time:.4f} seconds per generation")
        print(f"Average output length: {average_output_length:.2f} tokens")
        print(f"Average tokens per second (outputted): {tokens_per_second:.2f} tokens/sec")
    else:
        print("\nNo generations were performed.")

    # Generate and save the plot
    if inference_times_list and output_tokens_list:
        plt.figure(figsize=(10, 6))
        plt.scatter(output_tokens_list, inference_times_list, alpha=0.7)
        plt.title('Inference Time vs. Output Token Count')
        plt.xlabel('Output Token Count')
        plt.ylabel('Inference Time (seconds)')
        plt.grid(True)
        plot_file_path = os.path.join(save_path, 'inference_distribution.png')
        plt.savefig(plot_file_path)
        plt.close() # Close the plot to free up memory
        print(f"\nPlot saved to: {plot_file_path}")
    else:
        print("\nNot enough data to generate a plot.")


if __name__ == "__main__":
    test_generate()