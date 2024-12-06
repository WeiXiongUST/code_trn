from datasets import load_dataset
ds = load_dataset("feedbackagent/llama3_8b_math_train", split='train')
def filter_data0(example):
    if not example['my_solu'].startswith("<|begin_of_text|>"):
        return False
    return True
    
ds = ds.filter(filter_data0)

def filter_data1(example):
    if 'Reward score:1' in example['my_solu']:
        return False
    return True
ds = ds.filter(filter_data1)
def get_prob_ans(text):
    prom = text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[0]
    prom = prom.replace("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", "")
    tmp = text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[1]
    ans = tmp.split("<|eot_id|><|start_header_id|>user<|end_header_id|>")[0]
    return prom, ans

def generate_prompt(problem, reasoning_path):
    """
    Generates a simplified prompt for obtaining feedback on a reasoning path.

    Parameters:
        problem (str): The mathematical problem statement.
        reasoning_path (str): The step-by-step reasoning provided for solving the problem.

    Returns:
        str: A formatted prompt to elicit detailed feedback from a language model.
    """
    template = f"""
You are a mathematical reasoning assistant. Analyze the step-by-step reasoning provided for the problem below. Your feedback should:  
1. Identify errors or inconsistencies in logic or calculations.  
2. Suggest clearer or more rigorous explanations.  
3. Recommend corrections or alternative approaches if needed.  

---

**Problem**:  
{problem}

---

**Reasoning Path**:  
{reasoning_path}

---

**Your Feedback**:  
1. **Reflection**: Highlight errors or areas needing improvement.  
2. **Improvement Plan**: Suggest corrections or enhancements to improve the reasoning process.  

Provide your feedback in a structured format with references to specific steps.
"""
    return template

from collections import defaultdict

# 用 defaultdict 分组数据
grouped_data = defaultdict(list)
kk = 0
# 遍历数据并按 index 分组
for sample in ds:
    grouped_data[sample["idx"]].append(sample)
    kk += 1

all_data = []
all_idx = list(set(ds['idx']))
for idx in all_idx:
    pass_data2 = []
    pass_data1 = []
    for sample in grouped_data[idx]:
        prom, ans = get_prob_ans(sample['my_solu'])
        cnt = 0
        if ans not in pass_data1:
            
            all_data.append(
                    {"idx": sample['idx'], 'gt': sample['gt'], 'my_prompt': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + generate_prompt(prom, ans) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "old_solu": sample['my_solu']}
                )
            pass_data1.append(ans)
            cnt += 1
        if cnt >= 5:
            break

dict_data = {
"idx": [d['idx'] for d in all_data],
"gt": [d['gt'] for d in all_data],
"my_prompt": [d['my_prompt'] for d in all_data],
'old_solu': [d['old_solu'] for d in all_data],
}

from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(dict_data)
DatasetDict({"train": dataset}).push_to_hub("feedbackagent/baseprompt")
