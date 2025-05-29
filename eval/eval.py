from generate_2 import init, generate
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
import os
import asyncio

test_set = "data/test_SFT_con_sum.jsonl"

async def get_ROUGE_EVAL(test_path: str, lm_path: str, base_lm_path: str, icformer_path: str, batch_size: int = 10):
    """
    Evaluates a language model on summarization tasks using ROUGE scores and exact match for specific cases,
    running generation in asynchronous batches.

    Args:
        test_path (str): Path to the test dataset in JSONL format.
        lm_path (str): Path to the main language model.
        base_lm_path (str): Path to the base language model.
        icformer_path (str): Path to the ICFormer model.
        batch_size (int): The number of generation requests to run concurrently in a batch.
    """
    model = init(lm_path=lm_path, base_lm_path=base_lm_path, icformer_path=icformer_path)
    dataset = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    all_rouge_scores = {'rouge1': {'fmeasure': [], 'precision': [], 'recall': []},
                        'rouge2': {'fmeasure': [], 'precision': [], 'recall': []},
                        'rougeL': {'fmeasure': [], 'precision': [], 'recall': []},
                        'rougeLsum': {'fmeasure': [], 'precision': [], 'recall': []}}
    
    nothing_relevant_found_matches = 0
    total_nothing_relevant_found_cases = 0
    total_samples = 0

    results = [] # To store individual results for saving

    # Process dataset in batches

    #batches_to_run = len(dataset)
    batches_to_run = 6
    for i in tqdm(range(0, batches_to_run, batch_size), desc="Processing batches"):
        batch = dataset[i:i + batch_size]
        
        # Create a list of coroutines for the current batch
        tasks = [generate(model, row["prompt"], row["input"]) for row in batch]
        
        # Run all tasks in the batch concurrently
        batch_answers = await asyncio.gather(*tasks)

        for j, row in enumerate(batch):
            answer: str = batch_answers[j]
            ground_truth: str = row["answer"]

            total_samples += 1

            scores = {} # Initialize scores for each item

            if ground_truth.strip().lower() == "[nothing relevant found]":
                total_nothing_relevant_found_cases += 1
                if answer.strip().lower() == "[nothing relevant found]":
                    nothing_relevant_found_matches += 1
            else:
                scores = scorer.score(ground_truth, answer)
                for rouge_type in all_rouge_scores:
                    if rouge_type in scores:
                        all_rouge_scores[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)
                        all_rouge_scores[rouge_type]['precision'].append(scores[rouge_type].precision)
                        all_rouge_scores[rouge_type]['recall'].append(scores[rouge_type].recall)
            
            results.append({
                "sample_id": i + j, # Adjust sample ID for correct indexing
                "prompt": row["prompt"],
                "input": row["input"],
                "ground_truth": ground_truth,
                "model_output": answer,
                "rouge_scores": {k: {metric: getattr(v, metric) for metric in ['precision', 'recall', 'fmeasure']} for k, v in scores.items()} if scores else {}, # Ensure scores is not empty
                "is_nothing_relevant_found_case": (ground_truth.strip().lower() == "[nothing relevant found]"),
                "matched_nothing_relevant_found": (answer.strip().lower() == "[nothing relevant found]" and ground_truth.strip().lower() == "[nothing relevant found]")
            })

    # Calculate average ROUGE scores
    average_rouge_scores = {}
    for rouge_type, metrics in all_rouge_scores.items():
        average_rouge_scores[rouge_type] = {}
        for metric, values in metrics.items():
            average_rouge_scores[rouge_type][metric] = sum(values) / len(values) if values else 0

    # Calculate "nothing relevant found" accuracy
    nothing_relevant_found_accuracy = (nothing_relevant_found_matches / total_nothing_relevant_found_cases) if total_nothing_relevant_found_cases > 0 else 0

    # Prepare final evaluation results
    final_eval_results = {
        "average_rouge_scores": average_rouge_scores,
        "nothing_relevant_found_accuracy": nothing_relevant_found_accuracy,
        "total_nothing_relevant_found_cases": total_nothing_relevant_found_cases,
        "nothing_relevant_found_matches": nothing_relevant_found_matches,
        "total_samples": total_samples
    }

    # Print results
    print("\n--- Evaluation Results ---")
    print("Average ROUGE Scores:")
    for rouge_type, metrics in average_rouge_scores.items():
        print(f"  {rouge_type}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    print(f"\n'Nothing Relevant Found' Accuracy: {nothing_relevant_found_accuracy:.4f}")
    print(f"Total 'Nothing Relevant Found' Cases: {total_nothing_relevant_found_cases}")
    print(f"Matched 'Nothing Relevant Found' Instances: {nothing_relevant_found_matches}")
    print(f"Total Samples Processed: {total_samples}")

    # Save results to a JSON file
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, "evaluation_results.json") # You can change the filename
    with open(results_file_path, 'w', encoding='utf-8') as f:
        json.dump({"summary_results": final_eval_results, "detailed_results": results}, f, indent=4)
    print(f"\nDetailed results and summary saved to: {results_file_path}")

# Example of how to run the evaluation (you'll need to replace these with your actual paths)
async def main():
    await get_ROUGE_EVAL(
        test_path=test_set,
        lm_path="princeton-nlp/Llama-3-8B-ProLong-512k-Instruct", # <--- REPLACE WITH YOUR MODEL PATH
        base_lm_path="meta-llama/Meta-Llama-3.1-8B-Instruct", # <--- REPLACE WITH YOUR MODEL PATH
        icformer_path="output_c1/checkpoint-13092", # <--- REPLACE WITH YOUR MODEL PATH
        batch_size=10 # Run 10 requests concurrently
    )
if __name__ == "__main__":
    asyncio.run(main())