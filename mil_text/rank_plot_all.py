import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

datasets = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
            'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

algorithms = ['MI-Kernel', 'mi-Graph', 'miFV', 'mi-Net', 'MI-Net', 'MI-Net  \nwith DS', 'MI-Net  \nwith RC',
              'Res+pool', 'Res+pool\n-GCN', 'B-Res+pool\n-GCN (ours)']
my_pal = {'MI-Kernel': 'k', 'mi-Graph': 'gray', 'miFV': 'c', 'mi-Net': 'b',  'MI-Net': 'gold', 'MI-Net  \nwith DS': 'teal', 'MI-Net  \nwith RC': 'brown',
          'Res+pool': 'darkgreen', 'Res+pool\n-GCN': 'm', 'B-Res+pool\n-GCN (ours)': 'r'}

num_data_set = len(datasets)
num_alg = len(algorithms)

acc_matrix = np.loadtxt('rank_box_results.txt', delimiter=' ', usecols=range(num_alg))
print(acc_matrix)


rank = num_alg - np.argsort(np.argsort(acc_matrix, axis=1), axis=1)
print(rank)
for data_id_, data in enumerate(datasets):
    print('----------------------------------------------------------------')
    print(data + ', first: ' + algorithms[int(np.where(rank[data_id_]==1)[0])].strip() + ', second: ' + algorithms[int(np.where(rank[data_id_]==2)[0])].strip())

rank = rank.transpose()
# print(rank.shape)
rank_mean = np.mean(rank, axis=1)
print('Average rank')
print(rank_mean)
# rank_std = np.std(rank, axis=1)

rank_median = np.median(rank, axis=1)
print('Median rank')
print(rank_median)
order = np.argsort(rank_mean)

rank = rank[order][0: num_alg]
algorithms = [algorithms[idx] for idx in order]
algorithms = [algorithms[idx_new] for idx_new in np.arange(num_alg)]

print(algorithms)

rank_df = pd.concat([pd.DataFrame({algorithms[i]: rank[i, :]}) for i in range(num_alg)], axis=1)

# print(rank_df.head)

data_df = rank_df.melt(var_name='algorithm', value_name='Rank')
fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=75)
# plt.figure(figsize=(6, 9))
b = sns.boxplot(y="algorithm", x="Rank", data=data_df, showmeans=True, order=algorithms, whis=[0, 100],
                meanprops={"markerfacecolor":"black", "markeredgecolor":"black", "markersize":"50"}, palette=my_pal, linewidth=6)
# plt.ylabel("algorithm", size=18)
plt.xticks(ticks=np.arange(1, num_alg + 1, 1))
plt.xlabel("Rank", size=40)
# plt.plot(rank.mean(axis=1), np.arange(num_alg),  '--r*', lw=2)
b.tick_params(labelsize=30)
ax.set_ylabel('')
plt.tight_layout()
plt.show()