import numpy as np
import os
import multiprocessing as mp
import sys
import time
from res_pool_main import run_res_pool_one_dataset
from rFF_pool_main import run_rFF_pool_one_dataset
from ds_net_main import run_ds_net_one_dataset
indices = np.arange(5)


def run_all_methods(blank, index):

    if index == 3:
        pool_ = 'mean'
    else:
        pool_ = 'max'

    run_ds_net_one_dataset(blank, index, num_neib=0, pooling=pool_, r=0, alg_name='vanilla')

    if index == 0:
        k_gcn = 2
        k_bgcn = 2
        r_ = 1
    elif index == 1:
        k_gcn = 3
        k_bgcn = 3
        r_ = 10
    elif index == 2:
        k_gcn = 3
        k_bgcn = 3
        r_ = 5
    elif index == 3:
        k_gcn = 4
        k_bgcn = 4
        r_ = 10
    elif index == 4:
        k_gcn = 1
        k_bgcn = 1
        r_ = 10

    run_ds_net_one_dataset(blank, index, num_neib=k_gcn, pooling=pool_, r=0, alg_name='GCN')
    run_ds_net_one_dataset(blank, index, num_neib=k_bgcn, pooling=pool_, r=r_, alg_name='BGCN')


if __name__ == '__main__':

    pool = mp.Pool(processes=5)
    pool_results = [pool.apply_async(run_all_methods, (1, index)) for index in indices]
    pool.close()
    pool.join()
    for pr in pool_results:
        dict_results = pr.get()
