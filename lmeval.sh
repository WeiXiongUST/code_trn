#!/bin/bash

# lm_eval --model vllm \
#     --model_args pretrained=$1,tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.9,data_parallel_size={model_replicas} \
#     --tasks mmlu, \
#     --batch_size auto

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$1,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 32
