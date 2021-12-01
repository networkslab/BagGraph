import csv
import numpy as np

data_all = ['musk1', 'musk2', 'fox', 'tiger', 'elephant']
for data_index in range(len(data_all)):
    data_name = data_all[data_index]

    if data_index == 3:
        pool_ = 'mean'
    else:
        pool_ = 'max'

    k_gcn_list = [2, 3, 3, 4, 1]
    k_bgcn_list = [2, 3, 3, 4, 1]
    r_list = [1, 10, 5, 10, 10]

    prnt_str_head = ''
    prnt_str = ''

    alg_all = ['vanilla']
    k_all = [0]
    r_all = [0]

    alg_all.append('GCN')
    k_all.append(k_gcn_list[data_index])
    r_all.append(0)

    alg_all.append('BGCN')
    k_all.append(k_bgcn_list[data_index])
    r_all.append(r_list[data_index])

    for idx, alg_ in enumerate(alg_all):

        file_name = 'accuracy_' + data_name + '_ds_net_pool_num_neib_' + str(k_all[idx]) + '_' + pool_ + '_r_' + str(r_all[idx]) + '_' + alg_ + '.csv'

        raw_acc = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
        # print(raw_acc.shape)
        if raw_acc.ndim == 1:
            raw_acc = np.expand_dims(raw_acc, axis=1)

        num_alg = raw_acc.shape[1]

        for i in range(num_alg):
            # print('----------------------------------------------------------------------')
            # print('Algorithm : ' + algorithms[i])
            acc = np.squeeze(raw_acc[:, i])
            acc = np.reshape(acc, [10, 10])
            mu_ = np.mean(acc) * 100
            sigma_ = np.std(np.mean(acc, axis=0)) * 100
            new_str = '&' + "{:.1f}".format(mu_) + '$\pm$' + "{:.1f}".format(sigma_) + '       '
            prnt_str = prnt_str + new_str

            new_str_ = alg_ + '_k_' + str(k_all[idx]) + '_r_' + str(r_all[idx])
            prnt_str_head = prnt_str_head + new_str_.rjust(16) + ' '

    print(prnt_str_head)
    print(prnt_str)



