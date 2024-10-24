from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("pwork7/rlhflow_mixture_aimo_iter3")


no_template = ['pwork7/baseline_ultra_iter1', 'pwork7/baseline_ultra_iter2', 'pwork7/baseline_ultra_iter3', 'pwork7/rlhflow_mix_dart_iter3', 'pwork7/rlhflow_mix_dart_iter2', 'pwork7/rlhflow_mix_dart_iter1']

for name in no_template:
    tokenizer.push_to_hub(name)
