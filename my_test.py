
from datasets import load_dataset
ds1 = load_dataset("feedbackagent/test_reflection_eval1", split='train')
ds2 = load_dataset("feedbackagent/test_reflection_eval2", split='train')
ds3 = load_dataset("feedbackagent/test_reflection_eval3", split='train')

#ds3 = load_dataset("1231czx/corr_math3", split='train')
#ds4 = load_dataset("1231czx/corr_math4", split='train')

from datasets import concatenate_datasets, load_dataset
ds = concatenate_datasets([ds1, ds2, ds3])
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

keys = all_data[0].keys()
dict_data = {key: [d[key] for d in all_data] for key in keys}
from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(dict_data)
DatasetDict({"train": dataset}).push_to_hub("feedbackagent/test_reflection_eval_prompt")
