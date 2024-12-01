    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

  def modify_list(example):
    # Find the first two occurrences of -100
        lst = example['labels']
        first_end_idx = -1
        for i, num in enumerate(lst):
            if num != -100:
                first_end_idx = i - 1
                break

        # 如果整个列表都是 -100 或没有非 -100 的值
        if first_end_idx == -1:
            return lst

        # 找到第二次出现的 -100 的起始位置
        second_start_idx = -1
        for i in range(first_end_idx + 1, len(lst)):
            if lst[i] == -100:
                second_start_idx = i
                break

        # 如果没有找到第二次的 -100，则直接返回
        if second_start_idx == -1:
            return lst

        # 将两次出现之间的元素改为 -100
        for i in range(first_end_idx + 1, second_start_idx):
            lst[i] = -100

        example['labels'] = lst
        return example
    train_dataset = train_dataset.map(modify_list)
