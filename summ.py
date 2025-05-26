from datasets import load_dataset
from collections import defaultdict
from functools import reduce

# 所有模型的数据集
dataset_names = [
   "weqweasdas/qwen7b_base_with_score_passn",
    "weqweasdas/qwen7b_self_rewarding_sft_with_score_passn",
    "weqweasdas/qwen7b_grpo_ver2_step80_with_score_passn",
    #"weqweasdas/qwen7b_grpo_ver2_step80_with_score_passn_second_64", 
    #"weqweasdas/qwen7b_grpo_ver2_step200_with_score_passn", 
    #"weqweasdas/qwen7b_grpo_ver2_step300_with_score_passn",
]

# 指定 reference 数据集（只使用它的 500 条 prompt）
reference_dataset = "weqweasdas/qwen7b_base_with_score_passn"
ref_ds = load_dataset(reference_dataset, split="train")
reference_prompts = sorted(set([sample["prompt"] for sample in ref_ds]))  # 前500条，去重后排序
print(f"Loaded {len(reference_prompts)} unique prompts from reference dataset.")

# 存储每个模型：prompt -> is_correct
model_results = {}

for name in dataset_names:
    print(f"Processing {name}...")
    ds = load_dataset(name, split="train")

    # 构建 prompt -> is_correct（只针对 reference 的 prompt）
    prompt_correct = {}
    missing_prompts = 0
    for prompt in reference_prompts:
        matched = [s for s in ds if s["prompt"] == prompt]
        if not matched:
            prompt_correct[prompt] = False
            missing_prompts += 1
        else:
            score = matched[0]["score"]
            prompt_correct[prompt] = any(score)
    print(f"  Missing prompts in {name}: {missing_prompts}")
    model_results[name] = prompt_correct

# 构建统计表
stats_table = defaultdict(dict)  # prompt -> model -> is_correct
for prompt in reference_prompts:
    for model in dataset_names:
        stats_table[prompt][model] = model_results[model].get(prompt, False)

# 聚合分析
correct_by_model = {model: 0 for model in dataset_names}
union_correct = 0
intersection_correct = 0

for prompt in reference_prompts:
    results = [stats_table[prompt][model] for model in dataset_names]
    for model, correct in zip(dataset_names, results):
        if correct:
            correct_by_model[model] += 1
    if any(results):
        union_correct += 1
    if all(results):
        intersection_correct += 1

# 输出结果
print("\nCorrect counts per model:")
for model, count in correct_by_model.items():
    print(f"{model}: {count} / {len(reference_prompts)}")

print(f"\nUnion correct (any model correct): {union_correct}")
print(f"Intersection correct (all models correct): {intersection_correct}")
