import csv
import numpy as np
from scipy.stats import wilcoxon

alg_name = 'deepset'

k_gcn_list = [5]
k_bgcn_list = [5]
r_bgcn_list = [1]

prnt_str_head = ''
prnt_str = ''

alg_all = ['vanilla']
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

acc_all = []

for idx, alg_ in enumerate(alg_all):

    file_name = 'accuracy_' + alg_name + '_num_neib_' + str(k_all[idx]) + '_r_' + str(r_all[idx]) + '_' + alg_ + '.csv'

    raw_acc = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    acc_all.append(raw_acc)

    mu_ = np.mean(raw_acc) * 100
    sigma_ = np.std(raw_acc) * 100
    new_str = '&' + "{:.2f}".format(mu_) + '$\pm$' + "{:.2f}".format(sigma_) + ' '
    prnt_str = prnt_str + new_str.rjust(15)

    new_str_ = alg_ + '_k_' + str(k_all[idx]) + '_r_' + str(r_all[idx])
    prnt_str_head = prnt_str_head + new_str_.rjust(15) + ' '

print(prnt_str_head)
print(prnt_str)


print('----------------statistical test-----------')
_, p = wilcoxon(acc_all[0], acc_all[1], zero_method='wilcox', correction=False)
print('vanilla vs GCN')
print(p)
_, p = wilcoxon(acc_all[0], acc_all[2], zero_method='wilcox', correction=False)
print('vanilla vs BGCN')
print(p)
_, p = wilcoxon(acc_all[1], acc_all[2], zero_method='wilcox', correction=False)
print('GCN vs BGCN')
print(p)
