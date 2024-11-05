model=meta-llama/Llama-3.2-1B-Instruct
user_model=meta-llama/Llama-3.1-8B-Instruct
num_data=10

python ./setting_one/generate.py --model $model --user_model $user_model --output_dir output \
    --output_repo REFUEL --num_data $num_data