from datasets import load_dataset
ds = load_dataset("feedbackagent/reflection_n16", split='train')
all_data = []
for sample in ds:
    keys = sample.keys()
    pass_data = []
    for k in range(16):
        if sample['responses'][k] in pass_data:
            continue
        dict_sample = {key: sample[key] for key in keys}
        dict_sample.update({"reflection": sample['responses'][k]})
        all_data.append(dict_sample)
        pass_data.append(sample['responses'][k])

all_data1 = all_data[:299072]
all_data2 = all_data[299072:]

keys = all_data1[0].keys()
dict_data = {key: [d[key] for d in all_data1] for key in keys}
from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(dict_data)
DatasetDict({"train": dataset}).push_to_hub("feedbackagent/reflection_eval_prompt1")

keys = all_data2[0].keys()
dict_data = {key: [d[key] for d in all_data2] for key in keys}
from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(dict_data)
DatasetDict({"train": dataset}).push_to_hub("feedbackagent/reflection_eval_prompt2")
