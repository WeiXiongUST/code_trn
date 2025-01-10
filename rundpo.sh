accelerate launch --config_file ./training_configs/zero2_pf.yaml run_dpo.py \
    --model_name_or_path selfcorrexp/llama3_non_delete_rr40k_2e6_bz32_ep3 \
    --ref_model selfcorrexp/llama3_non_delete_rr40k_2e6_bz32_ep3 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --train_dir selfcorrexp/dpo_math1pair_and_augmath \
    --eval_dir selfcorrexp/dpo_math1pair_and_augmath \
    --learning_rate 2e-7 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 4 \
    --logging_steps 2 \
    --eval_steps 10000 \
    --output_dir=./mdpo_iter1_llama3_non_delete_lr2e7_bz32 \
    --report_to wandb \
    --save_strategy=steps \
    --save_steps=50 \
~                                                                                        
~                                                                                        
~                          
