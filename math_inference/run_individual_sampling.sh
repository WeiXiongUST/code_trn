python run_sampling.py \
    --completion_model_name_or_path pwork7/gemma7b_meta_math_3epoch_with_kn \
    --dataset_path 1231czx/ep2_step_data \
    --output_dir step_data \
    --tensor_parallel_size 1 \
    --num_gpus 8 \
    --local_rank $1 \
    --sampling_num 16 \
    --split $2 
