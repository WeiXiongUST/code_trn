import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

name = '/home/models/llama3-8b-sft_pxyyy_RLHFlow_mixture_bz128_lr2e5/checkpoint-1539'

output_name = 'pwork7/rlhflow_mixture_iter3'
tokenizer_name = name #'/home/wexiong_google_com/pwork/Online-RLHF/LLaMA3_iter1/checkpoint-300'

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
    #num_labels=1
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model.push_to_hub(output_name)
tokenizer.push_to_hub(output_name)
