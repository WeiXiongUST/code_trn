    def is_sublist(sublist, larger_list):
        return str(sublist)[1:-1] in str(larger_list)[1:-1]
    def set_minus_100(arr):
        inside_non_minus_100 = False  # 标记是否进入非 -100 区间
        for i in range(len(arr)):
            if arr[i] != -100:
                if not inside_non_minus_100:  # 第一次进入非 -100 区间
                    inside_non_minus_100 = True
                arr[i] = -100  # 修改为 -100
            elif inside_non_minus_100:  # 一旦重新遇到 -100，停止处理
                break
        return arr
        
  def modify_list(example):
        lst = example['labels']
        
        sequence = [4815, 3957, 856, 1455] # \n\nIs my most ...
        #sequence = [4800, 358, 690, 8881, 389]
        def find_sequence_start(lst_tmp, sequence):
            for i in range(len(lst_tmp) - len(sequence) + 1):
                if lst_tmp[i:i+len(sequence)] == sequence:
                    return i
            return -1
        start_index = find_sequence_start(lst, sequence)
        # [320, 9642, 477, 2360, 12106, 7566, 13], ' (Yes or No)? Yes.'

        #if is_sublist([320, 9642, 477, 2360, 12106, 7566, 13], lst):
            #return example
        # Update elements prior to the sequence
        if start_index != -1:
            lst[:start_index] = [-100] * start_index
        else:
            assert 1 == 0
        # [1102, 5084, 430, 279, 3766, 4320, 574, 3604, 4495, 13] " It seems that the previous answer was actually correct."

        if is_sublist([1102, 5084, 430, 279, 3766, 4320, 574, 3604, 4495, 13], example['labels']):
            lst = set_minus_100(lst)
        example['labels'] = lst
        return example
