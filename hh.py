def get_prompt(example):
    messages = [
                                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                                                    {"role": "user", "content": example['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}
                                                                            ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": full_prompt}
