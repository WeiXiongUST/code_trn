from datasets import load_dataset
ds1 = load_dataset("mathreward/base_iter1_data_collection_llama38b_math_1", split='train')
ds2 = load_dataset("mathreward/base_llama3_data_collection_8b_math_2", split='train')
#ds3 = load_dataset("1231czx/corr_math3", split='train')
#ds4 = load_dataset("1231czx/corr_math4", split='train')

#from datasets import concatenate_datasets, load_dataset

from datasets import concatenate_datasets, load_dataset
ds = concatenate_datasets([ds1, ds2])
def filter_data0(example):
    if not example['my_solu'].startswith("<|begin_of_text|>"):
        return False
    return True
ds = ds.filter(filter_data0)
def filter_data1(example):
    if not 'received a reward score:0' in example['my_solu']:
        return False
    return True

ds = ds.filter(filter_data1)

ds = ds.sort("idx")

import re

import re

# 示例字符串
text = "这是一些内容<|start_header_id|>需要提取的内容<|eot_id|>更多内容"

# 使用正则表达式提取内容
def get_re(text):
    pattern = r"<\|start_header_id\|>(user|assistant)<\|end_header_id\|>(.*?)(?=<\|start_header_id\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    # 如果找到匹配的内容
    if matches:
        role, extracted_content = matches[1]  # 提取第一个括号内的内容
        return extracted_content
    else:
        print("未找到匹配内容")
    


all_data = []
curr = 0
pass_data = []
cnt = 0
for sample in ds:
    if sample['idx'] != curr:
        curr = sample['idx']
        pass_data = []
        cnt = 0
    z = get_re(sample['my_solu'])
    if z not in pass_data:
        if cnt > 20:
            continue
        all_data.append(sample)
        cnt += 1
        pass_data.append(z)

    
