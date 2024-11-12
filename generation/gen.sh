# First approach: initialize 4 VLLM processes and split the prompt set to the 4 agents
# The generated samples will be stored at output_dir + local_index + ".jsonl

my_world_size=8 # how many gpu you use
my_K=16

prompt_dir=weqweasdas/ep1_2
output_dir=./gen_data

CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &

wait

prompt_dir=weqweasdas/ep1_2
output_dir=./gen_data

CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &

wait

prompt_dir=weqweasdas/ep1_2
output_dir=./gen_data

CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &

wait

prompt_dir=weqweasdas/ep1_2
output_dir=./gen_data

CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &

wait

prompt_dir=weqweasdas/ep1_2
output_dir=./gen_data

CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${my_K} --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &


