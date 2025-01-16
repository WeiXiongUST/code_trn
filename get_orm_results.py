from datasets import load_dataset
import numpy as np
import random
ds = load_dataset("1231czx/fixedbeta05_no_sft_llama3_sft_math_dpo_type1_7ktype2_8ktype4_7ktype3_ver2_100tmp10_vllmexp", split='train')#.select(range(40000, 50000))
tmp = load_dataset("1231czx/fixed_beta05_llama3_sft_math_type12_8ktype4_and_6ktype3_no_sft_loss100tmp10_vllmexp", split='train')

all_idxs= tmp['idx']
all_first_rewards = [sample['rewards'][0] for sample in tmp]
idx2rewards = dict(zip(all_idxs, all_first_rewards))

def proc(example):
    #return {"first_true_rewards": example['rewards'][0]} 

    return {'second_round_rewards': example['second_rewards'], "first_true_rewards": example['rewards'][0]}#idx2rewards[example['idx']]} 
    '''
    if len(example['messages']) < 5:
        return {"first_true_rewards": True}
    else:
        return {"first_true_rewards": False}
    '''
ds = ds.map(proc)

#def fil(example):
#    return example['proxy_label'] is not None
#ds = ds.filter(fil)
cnt1 = 0
cnt2 = 0
w2r = 0
r2w = 0
judge_r2r = 0
judge_r2w = 0
judge_w2r = 0
judge_w2w = 0

for sample in ds:
    if sample['first_true_rewards']:
        cnt1 += 1

    if sample['proxy_label']:
        if sample['first_true_rewards']:
            cnt2 += 1
            judge_r2r += 1
        else:
            judge_w2r += 1

    #else:
    elif sample['proxy_label'] == False:
        if sample['second_round_rewards'][0]:
            cnt2 += 1

        if sample['first_true_rewards'] and not sample['second_round_rewards'][0]:
            r2w += 1
        if not sample['first_true_rewards'] and sample['second_round_rewards'][0]:
            w2r += 1

        if sample['first_true_rewards']:
            judge_r2w += 1
        else:
            judge_w2w += 1
    else:
        if sample['first_true_rewards']:
            cnt2 += 1
        if not sample['first_true_rewards'] and sample['second_round_rewards'][0]:
            w2r += 1
            cnt2 += 1
print(cnt1, cnt2, r2w, w2r, judge_r2r, judge_r2w, judge_w2w, judge_w2r)
print('turn1 turn2 r2w w2r')
print(cnt1/len(ds), cnt2/len(ds), r2w/len(ds), w2r/len(ds))

print(judge_r2r / (judge_r2r + judge_r2w), judge_w2w/(judge_w2r + judge_w2w))

print(r2w/judge_r2w, w2r/judge_w2w)
