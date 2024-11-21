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
