import multiprocessing as mp
from deepset_main import run_deepset
from set_transformer_main import run_set_transformer


# def run_bgcn(blank, r):
#
#     run_set_transformer(num_neib=5, r=r, alg_name='BGCN')


if __name__ == '__main__':
    run_deepset(num_neib=0, r=0, alg_name='vanilla')
    run_deepset(num_neib=5, r=0, alg_name='GCN')
    run_deepset(num_neib=5, r=1, alg_name='BGCN')


