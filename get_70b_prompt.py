from datasets import load_dataset
ds = load_dataset("selfrew/base_llama3_8b_math_4", split='train')
def filter_data0(example):
    if not example['my_solu'][0].startswith("<|begin_of_text|>"):
        return False
    return True
    
ds = ds.filter(filter_data0)
import re
def process_data(example):
    text = example['my_solu'][0].split("ENDSIGNAL")[0]
    text = text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n\nReach max function call limit.", "")
    pattern = r"<\|start_header_id\|>(user|assistant)<\|end_header_id\|>(.*?)(?=<\|start_header_id\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    dialogue = []
    for role, content in matches:
        dialogue.append({"role": role, "content": content.strip().replace("<|eot_id|>", "") })
    return {"messages": dialogue}
def get_turn(example):
    return {"turn":len(example['messages'])}


ds1 = ds.map(process_data)
ds2 = ds1.map(get_turn)
def filter_data0(example):
    if len(example['messages']) not in [3, 5]:
        return False
    return True
ds2 = ds2.filter(filter_data0)

def filter_data(example):
    if example['turn'] == 3:
        return False
    return True

ds2 = ds2.filter(filter_data)

from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

def process_d0(example):
    
    return {"tmp": example['messages'][1]['content']}
ds2 = ds2.map(process_d0)

import pandas as pd
from datasets import Dataset

df = ds2.to_pandas()

# 按照 'prompt' 去重，保留第一次出现的记录
df_deduplicated = df.drop_duplicates(subset="tmp", keep="first")

# 转换回 Hugging Face 数据集格式
ds3 = Dataset.from_pandas(df_deduplicated)

ds3 = ds3.remove_columns(['tmp', 'solution', 'pred', 'my_solu'])
wrong = f"Your most recent response is not correct. Reflect on this feedback and improve your reasoning as follows::\n\n1.Reflection: Analyze the previous response based on the feedback. Identify specific errors, inconsistencies, or incomplete reasoning that led to the incorrect or suboptimal result.\n2.Improvement Plan: Clearly outline the issues found and propose concrete corrections or enhancements to the reasoning process.\n3. Revised Response: Generate a revised step by step reasoning path that addresses the identified issues while maintaining coherence and relevance to the original problem. Ensure the reasoning is logical and the conclusion aligns with the problem statement.\n4.Put the final answer within \\boxed{{}}."

def process_d(example):
    new_messages = []
    new_messages.append(example['messages'][0])
    new_messages.append(example['messages'][1])
    new_messages.append({"role": 'user', 'content': wrong})
    return {"my_prompt": tokenizer.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)}

ds4 = ds3.map(process_d)
ds4.push_to_hub("selfrew/correct_base_math4_4_70b")
