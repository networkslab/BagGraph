import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import optim
import seaborn as sns
from scipy import stats
import matplotlib as mpl
import scipy.sparse as sp
import networkx as nx
import time
import os
import glob
import csv
import torch
import torchvision
from models_all import *
from utils import *
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.neighbors import kneighbors_graph
mpl.rcParams['figure.dpi'] = 600
color = sns.color_palette()
#%matplotlib inline
pd.options.mode.chained_assignment = None  # default='warn'


def run_set_transformer_one_dataset(blank, data_index):
    datasets = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
            'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    data_name = datasets[data_index]
    print(data_name)

    torch.manual_seed(0)
    np.random.seed(0)

    num_neib = 3
    EPOCHS = 200
    MC_samples = 20

    num_trial = 10
    num_fold = 10

    lr_ = 1e-3

    log_dir = 'log_st/' + data_name

    check = os.path.isdir(log_dir)
    if not check:
        os.makedirs(log_dir)
        print("created folder : ", log_dir)
    else:
        print(log_dir, " folder already exists.")

    for f in os.listdir(log_dir):
        os.remove(os.path.join(log_dir, f))

    mat = loadmat('data/' + data_name + '_py.mat')  # load mat-file

    bag_ids = np.array(mat['bag_ids']).flatten()  # bags
    bag_features = np.array(mat['bag_features'])  # features
    labels = np.array(mat['labels']).flatten()  # labels

    df_features = pd.DataFrame(bag_features)
    df_bag_ids = pd.DataFrame(bag_ids, columns=['bag_ids'])

    df = pd.concat([df_bag_ids, df_features], axis=1)

    print(df_bag_ids.shape)
    print(df_features.shape)
    print(labels.shape)
    print(df.shape)

    # print(df.isna().sum().max())
    # print(df['bag_ids'].value_counts())

    x = df.iloc[:, :]
    print(x.shape)

    groups = x.groupby('bag_ids').mean()
    print(groups.shape)

    grouped_data = groups.values[:, :]
    y = labels
    print(y.shape)
    print(grouped_data.shape)

    scaled_features = x.copy()
    col_names = list(x)
    features = scaled_features[col_names[1:]]
    scaled_features[col_names[1:]] = features
    scaled_features.head()
    print(scaled_features.shape)

    groups = scaled_features.groupby('bag_ids')

    # Iterate over each group
    set_list = []
    for group_name, df_group in groups:
        single_set = []
        for row_index, row in df_group.iterrows():
            single_set.append(row[1:].values)
        set_list.append(single_set)

    print(len(set_list))

    target = set_list  # target = Set of sets, row = set,
    max_cols = max([len(row) for batch in target for row in batch])
    max_rows = max([len(batch) for batch in target])
    print(max_cols)
    print(max_rows)

    for i in range(len(set_list)):
        set_list[i] = np.array(set_list[i], dtype=float)

    y_ = y.copy().reshape(-1, 1)
    labels_copy = y.copy()
    print(labels_copy.shape)
    print(y_.shape)
    print(y.shape)

    def get_performance_test(trial_i, fold_j):
        torch.manual_seed(0)
        np.random.seed(0)

        y = y_

        models = [
            SetTransformer(in_features=200, num_heads=4, ln=False),
            STGCN(in_features=200, num_heads=4, ln=False)
        ]

        def weights_init(m):
            if isinstance(m, torch.nn.Linear) or isinstance(m, GraphConvolution):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        for model in models:
            model.apply(weights_init)

        # produce a split for training, validation and testing
        mat_fold = loadmat('data/fold/' + str(trial_i) + '/index' + str(fold_j) + '.mat')  # load mat-file

        idx_train = np.array(mat_fold['trainIndex']).flatten() - 1  # matlab index from 1, python from 0
        idx_test = np.array(mat_fold['testIndex']).flatten() - 1

        bce = nn.BCELoss()

        features = [torch.FloatTensor(set_) for set_ in set_list]
        labels = torch.FloatTensor(y)
        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)

        acc = []

        def train(epoch, adj_=None):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            if adj_ is not None:
                output, _ = model(features, adj_)
            else:
                output, _ = model(features)
            loss_train = bce(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            acc_train = accuracy(labels[idx_train], output[idx_train])
            acc_test = accuracy(labels[idx_test], output[idx_test])
            # if (epoch+1)%10 == 0:
            #     print('Epoch: {:04d}'.format(epoch+1),
            #                   'loss_train: {:.4f}'.format(loss_train.item()),
            #                   'acc_train: {:.4f}'.format(acc_train.item()),
            #                   'acc_test: {:.4f}'.format(acc_test.item()),
            #                   'time: {:.4f}s'.format(time.time() - t))

            return acc_test.item()

        for network in (models):

            # Model and optimizer
            model = network

            adj = None
            adj_norm = None

            if isinstance(model, (STGCN)):
                embedding = np.loadtxt(log_dir + '/decoding_set_transformer_' +
                                       data_name + '_trial_' + str(trial_i) + '_fold_' + str(fold_j), delimiter=",")

                A = kneighbors_graph(embedding, num_neib, mode='connectivity', include_self=True)
                G = nx.from_scipy_sparse_matrix(A)

                adj = nx.to_numpy_array(G)
                adj_norm = normalize(adj)
                adj_norm = torch.FloatTensor(adj_norm)

            optimizer = optim.Adam(model.parameters(), lr=lr_)

            # Train model
            t_total = time.time()
            for epoch in range(EPOCHS):
                if isinstance(model, SetTransformer):
                    value = train(epoch)
                else:
                    value = train(epoch, adj_norm)

            model.eval()
            with torch.no_grad():
                if isinstance(model, SetTransformer):
                    output, decoding = model(features)
                    np.savetxt(log_dir + '/decoding_set_transformer_' + data_name +
                               '_trial_' + str(trial_i) + '_fold_' + str(fold_j), decoding, delimiter=',')
                else:
                    output, _ = model(features, adj_norm)

            acc.append(accuracy(labels[idx_test], output[idx_test]))
            model.apply(weights_init)

            adj = None
            adj_norm = None
        return acc

    tests = []
    print('SetTransformer', 'STGCN')
    for i in range(num_trial):
        for j in range(num_fold):
            t = time.time()
            acc_all = get_performance_test(i+1, j+1)
            tests.append(acc_all)
            print('run : ' + str(num_fold*i+j+1) + ', accuracy : ' + str(np.array(acc_all)) + ', run time: {:.4f}s'.format(time.time() - t))

    tests = np.array(tests)
    df_test = pd.DataFrame(tests, columns=['SetTransformer', 'STGCN'])

    print(df_test.mean(axis=0))

    def get_performance_test_bayesian(trial_i, fold_j):
        torch.manual_seed(0)
        np.random.seed(0)

        y = y_

        models = [
            STGCN(in_features=200, num_heads=4, ln=False)
        ]

        def weights_init(m):
            if isinstance(m, torch.nn.Linear) or isinstance(m, GraphConvolution):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        for model in models:
            model.apply(weights_init)

        # produce a split for training, validation and testing
        mat_fold = loadmat('data/fold/' + str(trial_i) + '/index' + str(fold_j) + '.mat')  # load mat-file

        idx_train = np.array(mat_fold['trainIndex']).flatten() - 1  # matlab index from 1, python from 0
        idx_test = np.array(mat_fold['testIndex']).flatten() - 1

        bce = nn.BCELoss()

        features = [torch.FloatTensor(set_) for set_ in set_list]
        labels = torch.FloatTensor(y)
        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)

        acc = []

        def train_(epoch, adj_=None):
            t = time.time()
            model.train()
            optimizer.zero_grad()

            output, _ = model(features, adj_)

            loss_train = bce(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            acc_train = accuracy(labels[idx_train], output[idx_train])
            acc_test = accuracy(labels[idx_test], output[idx_test])
            # if (epoch+1)%10 == 0:
            #     print('Epoch: {:04d}'.format(epoch+1),
            #                   'loss_train: {:.4f}'.format(loss_train.item()),
            #                   'acc_train: {:.4f}'.format(acc_train.item()),
            #                   'acc_test: {:.4f}'.format(acc_test.item()),
            #                   'time: {:.4f}s'.format(time.time() - t))

            return acc_test.item(), output

        for network in (models):

            # Model and optimizer
            model = network

            adj = None
            adj_norm = None

            if isinstance(model, (STGCN)):
                embedding = np.loadtxt(log_dir + '/decoding_set_transformer_' +
                                       data_name + '_trial_' + str(trial_i) + '_fold_' + str(fold_j), delimiter=",")

                adj_np = MAP_inference(embedding, num_neib)

                adj_np_norm = normalize(adj_np)
                adj_norm = adj_np_norm
                adj_norm = torch.FloatTensor(adj_norm)

            optimizer = optim.Adam(model.parameters(), lr=lr_)

            # Train model
            output_ = 0.0
            t_total = time.time()
            for epoch in range(EPOCHS):
                value, output = train_(epoch, adj_norm)

                if epoch >= EPOCHS - MC_samples:
                    output_ += output

            output = output_ / np.float32(MC_samples)

            acc.append(accuracy(labels[idx_test], output[idx_test]))
            model.apply(weights_init)

            adj = None
            adj_norm = None
        return acc

    tests_bayesian = []
    print('B-STGCN')
    for i in range(num_trial):
        for j in range(num_fold):
            t = time.time()
            acc_bayes = get_performance_test_bayesian(i+1, j+1)
            tests_bayesian.append(acc_bayes)
            print('run : ' + str(num_fold*i+j+1) + ', accuracy : ' + str(np.array(acc_bayes)) + ', run time: {:.4f}s'.format(time.time() - t))

    tests_bayesian = np.array(tests_bayesian)
    df_test_bayesian = pd.DataFrame(tests_bayesian, columns=['B-STGCN'])

    df_concat = pd.concat([df_test, df_test_bayesian], axis=1)
    print(df_concat.mean(axis=0))
    df_concat.to_csv('accuracy_' + data_name + '_set_transformer.csv', index=False)

