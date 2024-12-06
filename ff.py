from datasets import load_dataset
ds1 = load_dataset("feedbackagent/llama3_8b_reflection1", split='train')
ds2 = load_dataset("feedbackagent/llama3_8b_reflection2", split='train')

from datasets import concatenate_datasets, load_dataset
ds = concatenate_datasets([ds1, ds2])
def filter_data0(example):
    if not example['my_solu'].startswith("**Your Feedback**:"):
        return False
    return True
ds = ds.filter(filter_data0)
def get_reflect(text):
    tmp = text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[1]
    ans = tmp.split("<|eot_id|><|start_header_id|>user<|end_header_id|>")[0]
    return ans
def get_old_prom(text):
    tmp = text.split("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")[0]
    return tmp + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

all_data = []
pass_reflect = []
for sample in ds:
    all_data.append(
            {"idx": sample['idx'], 'gt': sample['gt'], 'my_prompt': get_old_prom(sample['old_solu']) + get_reflect(sample['my_solu']) + "\n\nNow please generate a revised step-by-step response by addressing the identified issues while maintaining coherence with the problem statement. Put your final answer within \\boxed{}." + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "old_solu": sample['old_solu'], 'reflection': get_reflect(sample['my_solu'])}
        )
dict_data = {
"idx": [d['idx'] for d in all_data],
"gt": [d['gt'] for d in all_data],
"my_prompt": [d['my_prompt'] for d in all_data],
    "old_solu": [d['old_solu'] for d in all_data],
    'reflection': [d['reflection'] for d in all_data],
}
from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(dict_data)
DatasetDict({"train": dataset}).push_to_hub("feedbackagent/reflection_n16")
