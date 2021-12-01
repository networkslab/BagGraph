import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

plt.rcParams["figure.figsize"] = (12,6)

def super_impose(number: int, left: int = 150, bottom: int = 3500, width: int = 600, height: int = 950):
    file_name = 'log_ds/idx_test_trial_{}.txt'.format(number)
    test_indices = np.loadtxt(file_name, dtype=np.int)
    # print(sorted(test_indices))
    # print(len(test_indices))

    file_name = 'data/centroids_cartesian_10.csv'
    location = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(1, 3))
    print(location.shape)

    file_name = 'data/results-2016-election.csv'
    votes = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(1, 4))
    print(votes.shape)
    votes = votes/votes.sum(axis=1, keepdims=True)

    file_name = 'log_ds/output_deepset_bgcn_trial_{}_num_neib_5_r_1.txt'.format(number)
    preds_bgcn = np.loadtxt(open(file_name, "rb"), delimiter=",")

    file_name = 'log_ds/output_deepset_gcn_trial_{}_num_neib_5_r_0.txt'.format(number)
    preds_gcn = np.loadtxt(open(file_name, "rb"), delimiter=",")

    file_name = 'log_ds/output_deepset_trial_{}_num_neib_0_r_0.txt'.format(number)
    preds_deepset = np.loadtxt(open(file_name, "rb"), delimiter=",")

    d = {
            'x': location[:, 0],
            'y': -location[:, 1],
            'preds_deepset': preds_deepset,
            'preds_gcn': preds_gcn,
            'preds_bgcn': preds_bgcn,
            'vote': votes[:, 0],
            'is_test': np.zeros(votes.shape[0])
        }

    df = pd.DataFrame(data=d).convert_dtypes()

    for i in range(len(test_indices)):
        df.iloc[test_indices[i], -1] = 1

    df = df.loc[df['x'] > -3000]
    df = df.loc[df['y'] > 2000]

    df1 = pd.DataFrame(data=d).convert_dtypes()

    for i in range(len(test_indices)):
        df1.iloc[test_indices[i], -1] = 1

    df1 = df1.loc[df1['x'] >= left]
    df1 = df1.loc[df1['y'] >= bottom]
    df1 = df1.loc[df1['x'] <= left + width]
    df1 = df1.loc[df1['y'] <= bottom + height]

    df1['x'] = df1['x'] + 2500

    df1['x'] = (df1['x'] - 950)
    df1['y'] = (df1['y'] - 600)

    df_train = df1.loc[df1['is_test'] != 0]
    df_test = df1.loc[df1['is_test'] == 0]

    print(df_train.shape)
    print(df_test.shape)

    fig, axs = plt.subplots(2, 2)
    mynorm = plt.Normalize(vmin=0.3, vmax=0.7)


    ff = 16

    axs[0, 0].scatter(x=df['x'], y=df['y'], c=df['preds_deepset'], cmap='coolwarm', norm=mynorm, s=20)
    axs[0, 0].set_title('Deep Sets', fontsize=ff)
    axs[0, 1].scatter(x=df['x'], y=df['y'], c=df['preds_gcn'], cmap='coolwarm', norm=mynorm, s=20)
    axs[0, 1].set_title('DS-GCN', fontsize=ff)
    axs[1, 0].scatter(x=df['x'], y=df['y'], c=df['preds_bgcn'], cmap='coolwarm', norm=mynorm, s=20)
    axs[1, 0].set_title('B-DS-GCN', fontsize=ff)
    axs[1, 1].scatter(x=df['x'], y=df['y'], c=df['vote'], cmap='coolwarm', norm=mynorm, s=20)
    axs[1, 1].set_title('True Election Results', fontsize=ff)


    ## Plot train only
    axs[0, 0].scatter(x=df_train['x'], y=df_train['y'], c=df_train['preds_deepset'], cmap='coolwarm', norm=mynorm, s=20)
    axs[0, 1].scatter(x=df_train['x'], y=df_train['y'], c=df_train['preds_gcn'], cmap='coolwarm', norm=mynorm, s=20)
    axs[1, 0].scatter(x=df_train['x'], y=df_train['y'], c=df_train['preds_bgcn'], cmap='coolwarm', norm=mynorm, s=20)
    axs[1, 1].scatter(x=df_train['x'], y=df_train['y'], c=df_train['vote'], cmap='coolwarm', norm=mynorm, s=20)

    ## Plot test only
    axs[0, 0].scatter(x=df_test['x'], y=df_test['y'], c=df_test['preds_deepset'], cmap='coolwarm', norm=mynorm, s=20)
    axs[0, 1].scatter(x=df_test['x'], y=df_test['y'], c=df_test['preds_gcn'], cmap='coolwarm', norm=mynorm, s=20)
    axs[1, 0].scatter(x=df_test['x'], y=df_test['y'], c=df_test['preds_bgcn'], cmap='coolwarm', norm=mynorm, s=20)
    axs[1, 1].scatter(x=df_test['x'], y=df_test['y'], c=df_test['vote'], cmap='coolwarm', norm=mynorm, s=20)


    #             MID-WEST
    # specify the location of (left,bottom),width,height
    rect1 = mpatches.Rectangle(((275 + 1500)*1.5-1100, 2800), 835, 1150,
                              fill=False,
                              color="purple",
                              linewidth=2)
    rect2 = mpatches.Rectangle(((275 + 1500)*1.5-1100, 2800), 835, 1150,
                              fill=False,
                              color="purple",
                              linewidth=2)
    rect3 = mpatches.Rectangle(((275 + 1500)*1.5-1100, 2800), 835, 1150,
                              fill=False,
                              color="purple",
                              linewidth=2)
    rect4 = mpatches.Rectangle(((275 + 1500)*1.5-1100, 2800), 835, 1150,
                              fill=False,
                              color="purple",
                              linewidth=2)

    axs[0, 0].add_patch(rect1)
    axs[0, 1].add_patch(rect2)
    axs[1, 0].add_patch(rect3)
    axs[1, 1].add_patch(rect4)


    # specify the location of (left,bottom),width,height
    rect11 = mpatches.Rectangle((235, 3650), 385, 700,
                              fill=False,
                              color="purple",
                              linewidth=2)
    rect22 = mpatches.Rectangle((235, 3650), 385, 700,
                              fill=False,
                              color="purple",
                              linewidth=2)
    rect33 = mpatches.Rectangle((235, 3650), 385, 700,
                              fill=False,
                              color="purple",
                              linewidth=2)
    rect44 = mpatches.Rectangle((235, 3650), 385, 700,
                              fill=False,
                              color="purple",
                              linewidth=2)
    # facecolor="red")
    axs[0, 0].add_patch(rect11)
    axs[0, 1].add_patch(rect22)
    axs[1, 0].add_patch(rect33)
    axs[1, 1].add_patch(rect44)


    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # remove the x and y ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    plt.savefig('election.pdf')

    # plt.subplots_adjust(left=0.125,
    #                     bottom=0.1,
    #                     right=0.9,
    #                     top=0.9,
    #                     wspace=0.2,
    #                     hspace=0.35)


    plt.show()



if __name__ == "__main__":
    super_impose(8)
