#!/bin/bash

# lm_eval --model vllm \
#     --model_args pretrained=$1,tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.9,data_parallel_size={model_replicas} \
#     --tasks mmlu, \
#     --batch_size auto
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_IBEXT_DISABLE=1

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$1,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 4 \
    --verbosity WARNING

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$1,dtype=bfloat16 \
    --tasks ai2_arc \
    --num_fewshot 25 \
    --batch_size 4 \
    --verbosity WARNING

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$1,dtype=bfloat16 \
    --tasks truthfulqa_mc1,truthfulqa_mc2 \
    --num_fewshot 0 \
    --batch_size 4 \
    --verbosity WARNING
