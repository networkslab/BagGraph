import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from numpy import matlib
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch import optim
import torch.nn as nn
import matplotlib as mpl
import networkx as nx
import scipy.sparse as sp
import pickle as pk
import time
import torch
import sys
import glob
import os
from models_all_deepset import *
from sklearn.neighbors import kneighbors_graph
from scipy.stats import wilcoxon
from utils import *
from sklearn import preprocessing
from subprocess import call
mpl.rcParams['figure.dpi'] = 600
color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'


def run_deepset(num_neib, r, alg_name='vanilla'):
    num_trials = 100
    lr_ = 1e-3
    weight_decay_ = 1e-4
    EPOCHS = 200
    MC_samples = 5

    log_dir = 'log_ds'

    check = os.path.isdir(log_dir)
    if not check:
        os.makedirs(log_dir)
        print("created folder : ", log_dir)
    else:
        print(log_dir, " folder already exists.")

    file_name = 'data/centroids_cartesian_10.csv'
    location = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(1, 3))
    print(location.shape)

    file_name = 'data/results-2016-election.csv'
    votes = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(1, 4))
    votes = votes[:, [0, 2]]  # keeping republican and democrat
    votes = votes[:, [0]] / votes.sum(axis=1, keepdims=True)  # republican vote probability
    print(votes.shape)

    county_list = os.listdir('data/cleaned_features')
    # print(county_list)

    set_list = []

    for file_name_ in county_list:
        with open('data/cleaned_features/' + file_name_, 'rb') as f:
            set_ = pk.load(f)
        set_list.append(set_)

    def sample_dataset(n=100, seed=0):
        np.random.seed(seed)

        X = []
        for set_ in set_list:
            row_id = np.random.choice(set_.shape[0], n, replace=False)
            X.append(set_[row_id, :])

        X = np.array(X)
        X = (X - np.mean(X, axis=(0, 1), keepdims=True))/np.std(X, axis=(0, 1), keepdims=True)
        return X

    def get_performance_test(trial_i):
        torch.manual_seed(trial_i)
        np.random.seed(trial_i)

        indices = np.arange(0, votes.shape[0])
        np.random.shuffle(indices)
        num_train = int(0.025 * votes.shape[0])
        idx_train = indices[:num_train]
        idx_test = indices[num_train:]
        np.savetxt(log_dir + '/idx_test' + '_trial_' + str(trial_i) + '.txt', idx_test, delimiter=',')

        X = sample_dataset(n=100, seed=trial_i)
        X = np.array(X, dtype='float')

        features = torch.FloatTensor(X)
        labels = torch.FloatTensor(votes)

        acc = []

        if alg_name == 'vanilla':
            models = [
                DeepSet(in_features=94)
            ]
        elif alg_name == 'GCN':
            models = [
                DSGCN(in_features=94)
            ]

            A = kneighbors_graph(location, num_neib, mode='connectivity', include_self=True)
            G = nx.from_scipy_sparse_matrix(A)
            adj = nx.to_numpy_array(G)
            adj_norm = normalize(adj)
            adj_norm = torch.FloatTensor(adj_norm)

        def weight_reset(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        for model in models:
            model.apply(weight_reset)

        bce = nn.BCELoss()

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

            loss_test = bce(output[idx_test], labels[idx_test])
            acc_train = accuracy(labels[idx_train], output[idx_train])
            acc_test = accuracy(labels[idx_test], output[idx_test])
            # if (epoch+1) % 10 == 0:
            #     print('Epoch: {:04d}'.format(epoch+1),
            #                   'loss_train: {:.4f}'.format(loss_train.item()),
            #                   'loss_test: {:.4f}'.format(loss_test.item()),
            #                   'acc_train: {:.4f}'.format(acc_train.item()),
            #                   'acc_test: {:.4f}'.format(acc_test.item()),
            #                   'time: {:.4f}s'.format(time.time() - t))
            return acc_test.item()

        for network in (models):

            t_total = time.time()
            # Model and optimizer
            model = network

            no_decay = list()
            decay = list()
            for m in model.modules():
                if isinstance(m, torch.nn.Linear) or isinstance(m, GraphConvolution):
                    decay.append(m.weight)
                    no_decay.append(m.bias)

            optimizer = optim.Adam([{'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': weight_decay_}], lr=lr_)

            # Train model
            t_total = time.time()
            for epoch in range(EPOCHS):
                if isinstance(model, DeepSet):
                    value = train(epoch)
                else:
                    value = train(epoch, adj_norm)

            model.eval()
            with torch.no_grad():
                if isinstance(model, (DeepSet)):
                    output, decoding = model(features)
                    np.savetxt(log_dir + '/decoding_deepset' + '_trial_' + str(trial_i) + '_num_neib_' + str(num_neib) + '_r_' + str(r) + '.txt', decoding, delimiter=',')
                    np.savetxt(log_dir + '/output_deepset' + '_trial_' + str(trial_i) + '_num_neib_' + str(num_neib) + '_r_' + str(r) + '.txt', output, delimiter=',')

                if isinstance(model, (DSGCN)):
                    output, decoding = model(features, adj_norm)
                    np.savetxt(log_dir + '/decoding_deepset_gcn' + '_trial_' + str(trial_i) + '_num_neib_' + str(num_neib) + '_r_' + str(r) + '.txt', decoding, delimiter=',')
                    np.savetxt(log_dir + '/output_deepset_gcn' + '_trial_' + str(trial_i) + '_num_neib_' + str(num_neib) + '_r_' + str(r) + '.txt', output, delimiter=',')

            acc.append(accuracy(labels[idx_test], output[idx_test]))
            model.apply(weight_reset)
        return acc

    if alg_name == 'vanilla' or alg_name == 'GCN':
        tests = []
        if alg_name == 'vanilla':
            print('DeepSet')
        elif alg_name == 'GCN':
            print('DSGCN')
        for i in range(num_trials):
            t = time.time()
            acc_all = get_performance_test(i)
            tests.append(acc_all)
            print('run : ' + str(i+1) + ', accuracy : ' + str(np.array(acc_all)) + ', run time: {:.2f}s'.format(time.time() - t))

        tests = np.array(tests)

        if alg_name == 'vanilla':
            df_test = pd.DataFrame(tests, columns=['DeepSet'])
        elif alg_name == 'GCN':
            df_test = pd.DataFrame(tests, columns=['DSGCN'])

        print(df_test.mean(axis=0))

    def get_performance_test_bayesian(trial_i):
        torch.manual_seed(trial_i)
        np.random.seed(trial_i)

        indices = np.arange(0, votes.shape[0])
        np.random.shuffle(indices)
        num_train = int(0.025 * votes.shape[0])
        idx_train = indices[:num_train]
        idx_test = indices[num_train:]

        X = sample_dataset(n=100, seed=trial_i)
        X = np.array(X, dtype='float')

        features = torch.FloatTensor(X)
        labels = torch.FloatTensor(votes)

        acc = []

        if alg_name == 'BGCN':
            models = [
                DSGCN(in_features=94)
            ]

        def weight_reset(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        for model in models:
            model.apply(weight_reset)

        bce = nn.BCELoss()

        def train_(epoch, adj_=None):
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

            loss_test = bce(output[idx_test], labels[idx_test])
            acc_train = accuracy(labels[idx_train], output[idx_train])
            acc_test = accuracy(labels[idx_test], output[idx_test])
            return acc_test.item(), output

        for network in (models):

            t_total = time.time()
            # Model and optimizer
            model = network

            if isinstance(model, (DSGCN)):
                embedding = np.loadtxt(log_dir + '/decoding_deepset_gcn' + '_trial_' + str(trial_i) + '_num_neib_' + str(num_neib) + '_r_0.txt', delimiter=",")
                adj_np = MAP_inference(embedding, num_neib, r)
                adj_np_norm = normalize(adj_np)
                adj_norm = torch.FloatTensor(adj_np_norm)

            no_decay = list()
            decay = list()
            for m in model.modules():
                if isinstance(m, torch.nn.Linear) or isinstance(m, GraphConvolution):
                    decay.append(m.weight)
                    no_decay.append(m.bias)

            optimizer = optim.Adam([{'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': weight_decay_}], lr=lr_)

            # Train model
            t_total = time.time()
            output_ = 0.0
            for epoch in range(EPOCHS):
                if isinstance(model, DeepSet):
                    value, output = train_(epoch)
                else:
                    value, output = train_(epoch, adj_norm)

                if epoch >= EPOCHS - MC_samples:
                    output_ += output

            output = output_ / np.float32(MC_samples)
            np.savetxt(log_dir + '/output_deepset_bgcn' + '_trial_' + str(trial_i) + '_num_neib_' + str(num_neib) + '_r_' + str(r) + '.txt', output.detach().numpy(), delimiter=',')

            acc.append(accuracy(labels[idx_test], output[idx_test]))
            model.apply(weight_reset)
            adj_norm = None
        return acc

    if alg_name == 'BGCN':
        tests_bayesian = []
        print('B-DSGCN')
        for i in range(num_trials):
            t = time.time()
            acc_bayes = get_performance_test_bayesian(i)
            tests_bayesian.append(acc_bayes)
            print('run : ' + str(i+1) + ', accuracy : ' + str(np.array(acc_bayes)) + ', run time: {:.2f}s'.format(time.time() - t))

        tests_bayesian = np.array(tests_bayesian)

        df_test_bayesian = pd.DataFrame(tests_bayesian, columns=['B-DSGCN'])

        print(df_test_bayesian.mean(axis=0))

    if alg_name == 'BGCN':
        df_test_bayesian.to_csv('accuracy_deepset_num_neib_' + str(num_neib) + '_r_' + str(r) + '_' + alg_name + '.csv', index=False)

    else:
        df_test.to_csv('accuracy_deepset_num_neib_' + str(num_neib) + '_r_' + str(r) + '_' + alg_name + '.csv', index=False)
