import numpy as np
from scipy.stats import wilcoxon

alg = 'deepset'  # 'set_transformer'

quant = ['rmse', 'mae', 'mape']

for quant_ in quant:
    file_name = quant_ + '_' + alg + '.csv'
    result = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    print('Algorithm: ' + alg)
    print('---------------' + quant_ + ' mean and std. error---------------')
    print(np.round(result.mean(axis=0), 2))
    print(np.round(result.std(axis=0), 2))
    print('----------------------------------------------------')
    print(quant_ + ': statistical test')

    _, p = wilcoxon(result[:, 0], result[:, 1], zero_method='wilcox', correction=False)
    print('vanilla vs GCN')
    print(p)
    _, p = wilcoxon(result[:, 0], result[:, 2], zero_method='wilcox', correction=False)
    print('vanilla vs BGCN')
    print(p)
    _, p = wilcoxon(result[:, 1], result[:, 2], zero_method='wilcox', correction=False)
    print('GCN vs BGCN')
    print(p)
