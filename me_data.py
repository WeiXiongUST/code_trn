import os
import json
import datasets

from datasets import DatasetDict, Dataset
from huggingface_hub import HfApi
import numpy as np
# 获取当前路径下的所有文件夹
base_path = os.getcwd()
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# 目标子文件夹
subfolders = [ "math500","minerva_math", "olympiadbench","aime24", "amc23", ]

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    #data_splits = {}
    acc = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        jsonl_file = os.path.join(subfolder_path, "test.jsonl")
        
        if os.path.exists(jsonl_file):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        acc.append(np.mean([sam['score'] for sam in data]))
        
                #data_splits[subfolder] = Dataset.from_list(data)
    print(folder, acc)


print("所有数据集已处理完毕！")
