#  FP-Stereo
#  Copyright (C) 2020  RCSL, HKUST
#  GPL-3.0 License

import subprocess
from threading import Thread
import sys

# set the parameter options in the design space
height = 374
width = 1242

cost_func_list = [0, 1, 2, 3]
cost_window_list = [5, 7]

num_dir_list = [4]
max_parallel_disp_list = [(64, 4), (64, 8), (64, 16), (128, 8), (128, 16), (128, 32)]  # tuple: (disp_range, unroll_factor)
p1_p2_list = [(5, 36), (5, 56), (5, 26), (5, 46), (50, 1500), (80, 3200), (10, 600), (30, 1290)]

uniqueness_list = [0, 1]
lr_check_list = [0, 1, 2]
filter_win = 5
shd_window = 3

# the number of threads for parallel computation on multiple cores
num_threads = 10

def split_chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def threaded_function(thread_idx, subset):
    print('thread {} starting to launch.\n'.format(thread_idx))
    for parameter_list in subset:
        params_str = [str(param) for param in parameter_list]
        cmd_str = params_str
        #print(cmd_str)
        ret = subprocess.call(["python", "run_sdx.py"] + cmd_str)
    print('thread {} completely returned.\n'.format(thread_idx))


def compute_combinations():
    combinations = []
    for max_parallel_disp in max_parallel_disp_list:
        index = 0
        for cost_function in cost_func_list:
            for num_dir in num_dir_list:
                for window_size in cost_window_list:
                    max_disp = max_parallel_disp[0]
                    parallel_disp = max_parallel_disp[1]
                    p1_p2 = p1_p2_list[index]
                    index = index + 1
                    p1 = p1_p2[0]
                    p2 = p1_p2[1]
                    for uniqueness in uniqueness_list:
                        for lr_check in lr_check_list:
                            if num_dir==4:
                            	combinations.append([height,width,cost_function,window_size,max_disp,parallel_disp,num_dir,uniqueness,lr_check,filter_win,shd_window,p1,p2])
    return combinations


combinations = compute_combinations()

print('Total number of combinations: {}'.format(len(combinations)))
print(combinations)


# split all possible combinations to given threads
# each thread is dedicated to execute only a subset of parameters
combination_subsets = split_chunks(combinations, num_threads)

for thread_idx, subset in enumerate(combination_subsets):
    print('---------thread {} -------------'.format(thread_idx))
    print('parameters:', subset)
    Thread(target=threaded_function, args=(thread_idx, subset)).start()
