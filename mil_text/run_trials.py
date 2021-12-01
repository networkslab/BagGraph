import numpy as np
import os
import multiprocessing as mp
import sys
import time
from res_pool_main import run_res_pool_one_dataset
indices = np.arange(20)


def run_all_methods(blank, index):
    pool_ = 'mean'
    k_gcn_list = [2, 2, 2, 3, 3, 4, 3, 4, 2, 2, 3, 3, 3, 3, 4, 3, 2, 2, 4, 2]
    k_bgcn_list = [3, 3, 3, 4, 3, 3, 3, 2, 3, 2, 4, 3, 2, 3, 4, 4, 3, 2, 3, 4]
    r_list = [10, 5, 10, 5, 5, 10, 1, 1, 10, 10, 1, 10, 1, 10, 5, 10, 10, 5, 10, 5]

    run_res_pool_one_dataset(blank, index, num_neib=0, pooling=pool_, r=0, alg_name='vanilla')
    run_res_pool_one_dataset(blank, index, num_neib=k_gcn_list[index], pooling=pool_, r=0, alg_name='GCN')
    run_res_pool_one_dataset(blank, index, num_neib=k_bgcn_list[index], pooling=pool_, r=r_list[index], alg_name='BGCN')


if __name__ == '__main__':

    pool = mp.Pool(processes=7)
    pool_results = [pool.apply_async(run_all_methods, (1, index)) for index in indices]
    pool.close()
    pool.join()
    for pr in pool_results:
        dict_results = pr.get()
