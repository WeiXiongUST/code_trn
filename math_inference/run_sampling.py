import json
import re
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
import spacy
# from eval_gsm8k import is_number, extract_answer_number
# from eval_math import remove_boxed, process_results
#import util
import time
import os
#from util import is_number, extract_answer_number, remove_boxed, process_results
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline


@dataclass
class ScriptArguments:
    completion_model_name_or_path: str = field(default="", metadata={"help": "the completion model name or path locally or from huggingface."})
    dataset_path: str = field(default="", metadata={"help": "dataset path for generator data."})
    output_dir: str = field(default="mc_data",metadata={"help":"location to store the PRM data."})
    tensor_parallel_size: int = field(default=1,metadata={"help":""})
    num_gpus: int = field(default=2)
    local_rank:int = field(default=0)
    sampling_num:int = field(default=16)
    split:int = field(default=0)

if __name__ == "__main__":
    
    parser = HfArgumentParser((ScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]

    raw_dataset = load_dataset(args.dataset_path,split='train')
    dataset = []
    tmp_count = 0
    for i in tqdm(raw_dataset):
        tmp_count += 1
        if tmp_count > 10000:
            break
        dataset.append(i)
    print("------------")
    print("begin to preprocess the sampling data")
    print("------------")
    
    processed_dataset = []
    dataset = dataset[int((args.local_rank)/args.num_gpus * len(dataset)):int((args.local_rank+1)/args.num_gpus * len(dataset))]
    dataset = dataset[:10]
    print(len(dataset))
    ref_dataset = []
    for sample in tqdm(dataset):
        ref_dataset.append(sample)
        processed_dataset.append(sample['new_prompt'])
    
    sampling_params = SamplingParams(n=args.sampling_num, temperature=0.7, top_p=1, max_tokens=2048)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.completion_model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, dtype = "float16",gpu_memory_utilization=0.9,swap_space=100)
    print("------------")
    print("begin to label with markov process.")
    print("------------")
    
    format_prompt = processed_dataset
    count = 0
    completions_list = []
    for batch_num in tqdm(range(0,len(format_prompt),10000)):
        batch = format_prompt[batch_num:batch_num+10000]
        completions = llm.generate(batch, sampling_params)
        for j,output in enumerate(completions):
            prompt_temp = output.prompt
            generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
            completions_list.append(ref_dataset[count])
            if "The answer is" in processed_dataset[count]:
                #completions_list.append({"prompt":processed_dataset[count]['tmp_answer'],"completions":["" for i in range(len(generated_text))]})
                completions_list[count]['completions'] = ["" for i in range(len(generated_text))]
            else:
                #completions_list.append({"prompt":processed_dataset[count]['tmp_answer'],"completions":generated_text})
                completions_list[count]['completions'] = generated_text
            count += 1
        
        os.makedirs(args.output_dir,exist_ok=True)
        with open(f"{args.output_dir}/math_data_split{args.split}_{args.local_rank}.json",'w') as f:
            json.dump(completions_list,f,indent=4,ensure_ascii=False)
