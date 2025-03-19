import os
import json
import datasets

from datasets import DatasetDict, Dataset
from huggingface_hub import HfApi

# 获取当前路径下的所有文件夹
base_path = os.getcwd()
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# 目标子文件夹
subfolders = ["aime24", "math500", "amc23", "minerva_math", "olympiadbench"]

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    data_splits = {}

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        jsonl_file = os.path.join(subfolder_path, "test.jsonl")

        if os.path.exists(jsonl_file):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
                data_splits[subfolder] = Dataset.from_list(data)

    if data_splits:
        dataset = DatasetDict(data_splits)
        dataset_name = f"{mytestdpo}/{folder}"  # 目标数据集名称

        # 将数据集推送到 Hugging Face Hub
        dataset.push_to_hub(dataset_name)
        print(f"数据集 {dataset_name} 已成功上传！")

print("所有数据集已处理完毕！")
