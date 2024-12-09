from datasets import load_dataset
ds1 = load_dataset("feedbackagent/subset", split='train')
ds2 = load_dataset("feedbackagent/reflection_eval_prompt2", split='train')

from collections import defaultdict

# 用 defaultdict 分组数据
grouped_data1 = defaultdict(list)
kk = 0
# 遍历数据并按 index 分组
for sample in ds1:
    grouped_data1[sample["idx"]].append(sample)
    kk += 1
from collections import defaultdict

# 用 defaultdict 分组数据
grouped_data2 = defaultdict(list)
kk = 0
# 遍历数据并按 index 分组
for sample in ds2:
    grouped_data2[sample["idx"]].append(sample)
    kk += 1
all_data = []
all_idx = list(set(ds2['idx']))
all_generated_idx = list(set(ds1['idx']))

for idx in all_idx:
    if idx not in all_generated_idx:
        all_data.extend(grouped_data2[idx])
        continue
    tmp_data2 = grouped_data2[idx]
    tmp_data1 = grouped_data1[idx]
    
    all_anss = [z['reflection'] for z in tmp_data1]
    for sample in tmp_data2:
        if sample['reflection'] not in all_anss:
            all_data.append(sample)
        
      

output_dir = "feedbackagent/new_prompt2"


keys = all_data[0].keys()  

dict_data = {key: [d[key] for d in all_data] for key in keys}


dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub(output_dir)
