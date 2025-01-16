from datasets import load_dataset
import numpy as np
import random
input_dir = "tmpmodelsave/fixed_no_sft_type3_7k_step150_more_datatmp10"
ds = load_dataset(input_dir, split='train')#.select(range(5000))
output_dir = input_dir.replace("tmpmodelsave", "1231czx")
output_dir = output_dir + "_vllmexp"
import re
def process_data(example):
    #txt = example['my_solu'][0].split("<|eot_id|><|start_header_id|>user<|end_header_id|>")[0]
    #txt = txt.split(" \n\nIs my most")[0]

    split_text = example['my_solu'][0].split("<|eot_id|><|start_header_id|>assistant")
    if len(split_text) > 2:
        txt = split_text[0] + "<|eot_id|><|start_header_id|>assistant" + split_text[1] + "<|eot_id|><|start_header_id|>assistant\n\n"


    if '(Yes or No)? Yes' in example['my_solu'][0]:
        label = True
        #txt = txt + f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nNo further modification is needed" + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif '(Yes or No)? No' in example['my_solu'][0]:
        label = False
        #txt = txt + f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSince your initial response is self-evaluated as incorrect, there might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Put your final answer within \\boxed{{}}." + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        label = None
    return {"my_prompt": txt, 'proxy_reward': label}

ds = ds.map(process_data)
print(ds[1])
ds.push_to_hub(output_dir)
