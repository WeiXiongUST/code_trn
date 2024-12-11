from datasets import load_dataset
ds1 = load_dataset("feedbackagent/test_reflection_eval_completion1_with_rewards", split='train')#ds3 = load_dataset("1231czx/corr_math3", split='train')
ds2 = load_dataset("feedbackagent/test_reflection_eval_completion2_with_rewards", split='train')#ds3 = load_dataset("1231czx/corr_math3", split='train')
ds3 = load_dataset("feedbackagent/test_reflection_eval_completion3_with_rewards", split='train')#ds3 = load_dataset("1231czx/corr_math3", split='train')
ds4 = load_dataset("feedbackagent/test_reflection_eval_completion4_with_rewards", split='train')#ds3 = load_dataset("1231czx/corr_math3", split='train')

#ds4 = load_dataset("1231czx/corr_math4", split='train')

from datasets import concatenate_datasets, load_dataset
ds = concatenate_datasets([ds1, ds2, ds3])

import numpy as np
def process_example(example):
    return {"accu": np.mean(example['rewards'])}
ds = ds.map(process_example)
from collections import defaultdict

# 加载一个示例数据集
# 创建一个字典用于分组
grouped_data = defaultdict(list)

# 遍历数据集，根据 idx 分组
for example in ds:
    grouped_data[example["idx"]].append(example)
    
for idx in grouped_data:
    grouped_data[idx] = sorted(grouped_data[idx], key=lambda x: x["response"])

kk = 0
all_record = []
all_data = []
for group_idx, group in grouped_data.items():
    ds1 = group #ds.filter(lambda ex: ex["idx"] == k)
    
    #ds1 = ds1.sort("reflection")
    z = 0
    z_corr = 0
    curr = ds1[0]['response']
    #old = ds1[0]['response']
    record = []
    tmp_reflection = []
    for sample in ds1:
        if sample['response'] != curr:
            all_record.append(record)
            if np.max(record) > 0: 
                all_data.append({'problem': sample['problem'], 'response': curr, 'label': record, 'answers': tmp_reflection})
            record = []
            tmp_reflection = []
            curr = sample['response']
            
        record.append(sample['accu'])
        tmp_reflection.append(sample['reflection'])

    kk += 1 
    if (kk+1)%100 == 0:
        print(kk)

keys = all_data[0].keys()
dict_data = {key: [d[key] for d in all_data] for key in keys}
from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(dict_data)
DatasetDict({"train": dataset}).push_to_hub("feedbackagent/test_set_short")
