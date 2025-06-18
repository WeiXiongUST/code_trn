# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from huggingface_hub import create_repo

# Import your custom verification utilities
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify.errors import TimeoutException


# ==============================================================================
# 2. CORE FUNCTIONS
# ==============================================================================

def merge_datasets(cot_dataset: Dataset, ppo_dataset: Dataset) -> Dataset:
    """
    Adds 'solution' from a CoT dataset to a PPO dataset by matching on the 'problem' field.
    """
    print("Building a lookup dictionary from the CoT dataset...")
    problem_to_solution = {example["problem"]: example["solution"] for example in tqdm(cot_dataset)}

    print("Matching problems and adding solutions to the PPO dataset...")
    new_examples = []
    for example in tqdm(ppo_dataset):
        problem = example["problem"]
        solution = problem_to_solution.get(problem)
        if solution:  # Only keep examples where a solution was found
            new_example = dict(example)
            new_example["solution"] = solution
            new_examples.append(new_example)

    print(f"Found matches for {len(new_examples)} / {len(ppo_dataset)} examples.")
    return Dataset.from_list(new_examples)

def verify_and_filter_solutions(dataset: Dataset) -> Dataset:
    """
    Scores each example's solution against its ground truth and filters for correct ones (score=1.0).
    """
    # Define the scoring function to be used with .map()
    def compute_score_for_example(example):
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        score = 0.0
        ground_truth = example['reward_model']['ground_truth']
        model_output = example['solution']
        
        # Wrap the ground truth in \boxed{} format for verification
        ground_truth_boxed = f"\\boxed{{{ground_truth}}}"
        
        try:
            score, _ = verify_func(gold_texts=[ground_truth_boxed], pred_texts=[model_output])
        except TimeoutException:
            score = 0.0  # Or a specific score for timeouts
        except Exception:
            score = 0.0  # Treat other errors as incorrect

        return {"score": score}

    print("Scoring all solutions in the dataset...")
    # Use .map() for efficient, parallelized scoring
    scored_dataset = dataset.map(compute_score_for_example, num_proc=4) # Adjust num_proc as needed

    scores = scored_dataset["score"]
    mean_score = np.mean(scores)
    print(f"Mean score of solutions before filtering: {mean_score:.4f}")

    print("Filtering dataset to keep only correct solutions (score = 1.0)...")
    # Use .filter() for an efficient, parallelized filter
    filtered_ds = scored_dataset.filter(lambda example: example['score'] == 1.0, num_proc=4)

    return filtered_ds

def deduplicate_by_prompt(dataset: Dataset, column: str = "problem") -> Dataset:
    """
    Removes duplicate examples from a dataset based on a specific column.
    Keeps the first occurrence of each unique prompt.
    """
    print(f"Deduplicating dataset based on the '{column}' column...")
    seen_prompts = set()
    indices_to_keep = []

    for idx, example in enumerate(tqdm(dataset)):
        prompt = example[column]
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            indices_to_keep.append(idx)

    print(f"Found {len(seen_prompts)} unique prompts.")
    deduplicated_ds = dataset.select(indices_to_keep)
    return deduplicated_ds

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Define dataset names
    cot_dataset_name = "AI-MO/NuminaMath-CoT"
    ppo_dataset_name = "RLHFlow/numia_prompt_ppo"

    # --- Step 1: Load original datasets ---
    print(f"Loading '{cot_dataset_name}'...")
    ds_cot = load_dataset(cot_dataset_name, split="train")
    print(f"Loading '{ppo_dataset_name}'...")
    ds_ppo = load_dataset(ppo_dataset_name, split="train")
    
    # --- Step 2: Merge datasets to add solutions ---
    ds_merged = merge_datasets(ds_cot, ds_ppo)
    print(f"Original merged dataset size: {len(ds_merged)}")

    # --- Step 3: Verify solutions and filter correct ones ---
    ds_filtered_correct = verify_and_filter_solutions(ds_merged)
    print(f"Dataset size after filtering for correct solutions: {len(ds_filtered_correct)}")

    # --- Step 4: Remove duplicate prompts ---
    final_dataset = deduplicate_by_prompt(ds_filtered_correct, column="problem")
    print(f"Final dataset size after removing duplicate prompts: {len(final_dataset)}")

    # --- Step 5 (Optional): Push to Hub ---
    # print("Pushing the final dataset to the Hugging Face Hub...")
    # hub_repo_name = "numia_prompt_ppo_verified_dedup"
    # create_repo(hub_repo_name, repo_type="dataset", exist_ok=True)
    # final_dataset.push_to_hub(hub_repo_name)
    # print("Done!")
