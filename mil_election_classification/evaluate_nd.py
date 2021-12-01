import csv
import numpy as np
from scipy.stats import wilcoxon

alg_name = 'deepset'
log_dir = 'log_ds'
num_trials = 100

file_name = 'data/results-2016-election.csv'
votes = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(1, 4))
votes = votes[:, [0, 2]]  # keeping republican and democrat
votes = votes[:, [0]] / votes.sum(axis=1, keepdims=True)  # republican vote probability
votes = np.squeeze(votes)

k_gcn_list = [5]
k_bgcn_list = [5]
r_bgcn_list = [1]

alg_all = ['Vanilla']
k_all = [0]
r_all = [0]

for k in k_gcn_list:
    alg_all.append('GCN')
    k_all.append(k)
    r_all.append(0)

for k_ in k_bgcn_list:
    for r_ in r_bgcn_list:
        alg_all.append('BGCN')
        k_all.append(k_)
        r_all.append(r_)

nd_all = np.zeros([num_trials, len(alg_all)])

for idx, alg_ in enumerate(alg_all):

    for i in range(num_trials):

        idx_test = np.squeeze(np.loadtxt(log_dir + '/idx_test_trial_' + str(i) + '.txt')).astype(int)

        if idx == 0:
            filename = log_dir + '/output_' + alg_name + '_trial_' + str(i) + '_num_neib_' + str(k_all[idx]) + '_r_' + str(r_all[idx]) + '.txt'
        else:
            filename = log_dir + '/output_' + alg_name + '_' + alg_.lower() + '_trial_' + str(i) + '_num_neib_' + str(k_all[idx]) + '_r_' + str(r_all[idx]) + '.txt'

        output = np.squeeze(np.loadtxt(filename))

        err = np.abs(votes - output)
        nd_all[i, idx] = np.mean(err[idx_test])/np.mean(np.abs(output[idx_test])) * 100


print(nd_all)

print('mean ND')
print(nd_all.mean(axis=0))
print('std. error of ND')
print(nd_all.std(axis=0))

print('----------------statistical test-----------')
_, p = wilcoxon(nd_all[0], nd_all[1], zero_method='wilcox', correction=False)
print('vanilla vs GCN')
print(p)
_, p = wilcoxon(nd_all[0], nd_all[2], zero_method='wilcox', correction=False)
print('vanilla vs BGCN')
print(p)
_, p = wilcoxon(nd_all[1], nd_all[2], zero_method='wilcox', correction=False)
print('GCN vs BGCN')
print(p)
