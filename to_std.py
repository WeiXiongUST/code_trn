from datasets import load_dataset
import numpy as np
ds = load_dataset("weqweasdas/8b_corr_math", split='train')


import re
def process_data(example):
    conversation = example['my_solu'].replace("<|begin_of_text|>", "")
    if "ENDSIGNAL" in conversation:
        conversation = conversation.split("ENDSIGNAL")[0]
    #user_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>"
    #assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>"

    pattern = r"<\|start_header_id\|>(user|assistant)<\|end_header_id\|>(.*?)(?=<\|start_header_id\|>|$)"
    matches = re.findall(pattern, conversation, re.DOTALL)
    
    dialogue = []
    for role, content in matches:
        dialogue.append({"role": role, "content": content.strip()})
    
    return {"messages": dialogue}


def filter_data_0(example):
    if len(example['messages']) < 2:
        return False
    if 'Reach max function' in example['my_solu']:
        return False
    return True    

def process_data2(example):
    self_correct = False
    ans_correct = False
    if example['messages'][-1]['role'] == 'user':
        if 'Reward score:1' in example['messages'][-1]['content'] and len(example['messages']) > 3:
            self_correct = True
        if 'Reward score:1' in example['messages'][-1]['content']:
            ans_correct = True
    return {"self_correct": self_correct, "ans_correct": ans_correct}

ds = ds.map(process_data, num_proc=8)
ds = ds.filter(filter_data_0, num_proc=8)
ds = ds.map(process_data2, num_proc=8)

def filter_data_self_correct(example):
    return example['self_correct']
def filter_data_ans_correct(example):
    return example['ans_correct']

new_ds1 = ds.filter(filter_data_self_correct)
new_ds2 = ds.filter(filter_data_ans_correct)
print(new_ds1, new_ds2)

from collections import Counter

# 假设你的数组如下
array1 = new_ds1['idx']
array2 = new_ds2['idx']
freq1 = Counter(array1)
freq2 = Counter(array2)

result = [
    element for element in list(range(7500))
    if freq1.get(element, 0) > 0
]
print(len(result))

'''
result = [
    element for element in list(range(7500))
    if freq1.get(element, 0) <= 3 and freq2.get(element, 0) <= 35
]
print(len(result))

result2 = [
    element for element in list(range(7500))
    if freq2.get(element, 0) < 1
]
print(len(result2))
'''
