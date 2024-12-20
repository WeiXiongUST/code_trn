def inspect_artifact_freq(ds, n_total=10000, artifact_mode='add_prefix'):
  artifact_count = 0
  for i in range(n_total):
    messages = ds[i]['messages']
    delimiters = r'\[CONTEXT\]|\[RESPONSE A\]|\[RESPONSE B\]'
    # Split the string
    result = re.split(delimiters, messages[0]['content'])
    # Remove empty strings that may result from consecutive delimiters or delimiters at the start/end
    result = [x for x in result if x]
    assert len(result) == 3
    label = messages[1]['content']
    assert label in ['A', 'B', 'Same']

    if label == 'A':
      win_res = result[1]
    else:
      win_res = result[2]
    if artifact_mode == 'bold_face':
      if win_res.startswith("**"):
        artifact_count += 1
    elif artifact_mode == 'add_emoji':
      if win_res.endswith("😀"):
        artifact_count += 1

  artifact_freq = artifact_count/n_total
  return artifact_freq


            full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a reasoning assistant tasked with solving math problems using chain-of-thought (CoT) reasoning. Your goal is to iteratively improve your responses based on feedback provided as a reward score (ranging from 0 to 1, where 1 indicates a perfect response). This process will repeat multiple times until the reasoning achieves a satisfactory reward score.\n\nProblem:\n{example['question']}\nPlease reason step by step, generate a complete CoT reasoning path, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                exec_result = f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nReward score:{rm_score}\n1.Reflect on the most recent response, considering the reward score (ranging from 0 to 1, where 1 indicates a perfect response).\n2.Identify errors, inconsistencies, or incomplete reasoning in the latest CoT response. Be specific about what went wrong and propose concrete improvements.\n3.Generate a revised CoT response by addressing the identified issues while maintaining coherence with the problem statement and prior context.\n4.Put your final answer within \\boxed{{}}."#"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
