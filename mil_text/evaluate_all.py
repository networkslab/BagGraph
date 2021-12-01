import csv
import numpy as np

datasets = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
            'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
for data_index in range(len(datasets)):
    data_name = datasets[data_index]
    print(data_name.ljust(25) + str(data_index+1) +  '  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    pool_ = 'mean'

    alg_list = ['vanilla', 'GCN', 'BGCN']

    k_list = [2, 3, 4]
    r_list = [1, 5, 10]

    prnt_str_head = ''
    prnt_str = ''

    alg_all = ['vanilla']
    k_all = [0]
    r_all = [0]

    for k_ in k_list:
        alg_all.append('GCN')
        k_all.append(k_)
        r_all.append(0)

    for k_ in k_list:
        for r_ in r_list:
            alg_all.append('BGCN')
            k_all.append(k_)
            r_all.append(r_)

    for idx, alg_ in enumerate(alg_all):

        file_name = 'accuracy_' + data_name + '_res_pool_num_neib_' + str(k_all[idx]) + '_' + pool_ + '_r_' + str(r_all[idx]) + '_' + alg_ + '.csv'

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
            new_str = '&' + "{:.1f}".format(mu_) + '$\pm$' + "{:.1f}".format(sigma_)
            prnt_str = prnt_str + new_str.rjust(16) + ' '

            new_str_ = alg_ + '_k_' + str(k_all[idx]) + '_r_' + str(r_all[idx])
            prnt_str_head = prnt_str_head + new_str_.rjust(16) + ' '

    print(prnt_str_head)
    print(prnt_str)



