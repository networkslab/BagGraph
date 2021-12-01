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
import scipy.sparse as sp
import time
import torch
import sys
import glob
import os
from models_all_set_transformer import *
from scipy.stats import wilcoxon
from utils import *
from sklearn import preprocessing
from subprocess import call
mpl.rcParams['figure.dpi'] = 600
color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'
torch.manual_seed(0)
np.random.seed(0)

num_trials = 100
lr_ = 5e-4
weight_decay_ = 1e-3
EPOCHS = 500
MC_samples = 20

adj_orig = np.loadtxt('data/adj_nbhd.txt', dtype='float', delimiter=',')
print(adj_orig)
num_neib = int(np.sum(adj_orig)/adj_orig.shape[0]) - 1
print(num_neib)

files = glob.glob('*.pkl')
for file in files:
    os.remove(file)
log_dir = 'log_st'

check = os.path.isdir(log_dir)
if not check:
    os.makedirs(log_dir)
    print("created folder : ", log_dir)
else:
    print(log_dir, " folder already exists.")

for f in os.listdir(log_dir):
    os.remove(os.path.join(log_dir, f))

train_df = pd.read_csv('data/neighbourhood_data.csv')
districts = pd.read_csv('data/districts_data_cleaned.csv')
names = list(districts['name'].unique())
train_df = train_df[train_df['neighbourhood'].isin(names)]
counts = train_df['neighbourhood'].value_counts()
count_list = counts[counts > 50].index.tolist()
train_df = train_df[train_df['neighbourhood'].isin(count_list)]


train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
train_df = train_df.astype({'bathrooms': 'int64'})
train_df = train_df.astype({'bedrooms': 'int64'})
train_df = train_df.astype({'interest_level': 'category'})
train_df = train_df.astype({'num_photos': 'int64'})
train_df = train_df.astype({'num_features': 'int64'})
train_df = train_df.astype({'num_description_words': 'int64'})
train_df = train_df.astype({'created_month': 'category'})
train_df = train_df.astype({'created_day': 'category'})
train_df = train_df.astype({'neighbourhood': 'str'})
train_df = train_df.astype({'price': 'float64'})
non_standardized = train_df.copy()
train_df['price'] = (train_df['price'] - train_df['price'].mean()) / train_df['price'].std()

train_df['interest_level'] = pd.Categorical(train_df['interest_level'], categories=train_df['interest_level'].unique()).codes
train_df['neighbourhood'] = pd.Categorical(train_df['neighbourhood'], categories=train_df['neighbourhood'].unique()).codes


def train_Set(n=25, seed=None):
    train_samples = train_df.groupby('neighbourhood').apply(pd.DataFrame.sample, n, replace = False, random_state=seed)
    train_samples = train_samples.astype({'price': 'float64'})
    features = []
    for i, hood in enumerate(train_samples['neighbourhood'].unique()):
        get_boro = train_samples[train_samples['neighbourhood'] == hood]
        sample = get_boro.to_numpy()
        features.append(sample)
    return np.array(features)


non_standardized['price'] = non_standardized['price'].clip(upper = np.percentile(non_standardized['price'].values, 95))
df = train_df[['price', 'neighbourhood']]
mean = df.groupby('neighbourhood').mean().values


def get_performance_test(trial_i):
    torch.manual_seed(trial_i)
    np.random.seed(trial_i)

    indices = np.arange(0, mean.shape[0])
    np.random.shuffle(indices)
    idx_train = indices[:55]
    idx_test = indices[55:]

    X = train_Set(n=25, seed=trial_i)
    X = np.array(X, dtype='float')

    features = torch.FloatTensor(X)
    labels = torch.FloatTensor(mean)

    mserr = []
    maerr = []
    mperr = []

    adj_norm = normalize(adj_orig)
    adj_norm = torch.FloatTensor(adj_norm)

    models = [
        SetTransformer(in_features=10),
        STGCN(in_features=10)
    ]

    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    for model in models:
        model.apply(weight_reset)

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    def train(epoch, adj_=None):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if adj_ is not None:
            output, _ = model(features, adj_)
        else:
            output, _ = model(features)
        loss_train = mse(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_test = mse(output[idx_test], labels[idx_test])
        return loss_test.item()

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
            if isinstance(model, SetTransformer):
                value = train(epoch)
            else:
                value = train(epoch, adj_norm)

        model.eval()
        with torch.no_grad():
            if isinstance(model, (SetTransformer)):
                output, decoding = model(features)
                np.savetxt(log_dir + '/decoding_set_transformer' + '_trial_' + str(trial_i), decoding, delimiter=',')

            if isinstance(model, (STGCN)):
                output, decoding = model(features, adj_norm)
                np.savetxt(log_dir + '/decoding_set_transformer_gcn' + '_trial_' + str(trial_i), decoding, delimiter=',')

        maerr.append(non_standardized['price'].std() * (mae(output[idx_test], labels[idx_test]).detach().cpu().numpy()))
        mserr.append(non_standardized['price'].std() * np.sqrt(mse(output[idx_test], labels[idx_test]).detach().numpy()))

        target = (non_standardized['price'].std() * labels[idx_test]).detach().cpu().numpy() + non_standardized['price'].mean()
        pred = (non_standardized['price'].std() * output[idx_test].T).detach().cpu().numpy() + non_standardized['price'].mean()

        mperr.append(100 * np.mean(np.abs((target - pred) / target)))
        model.apply(weight_reset)
    return mserr, maerr, mperr


mse = []
mae = []
mpe = []
print('SetTransformer', 'STGCN')
for i in range(num_trials):
    t = time.time()
    mserr, maerr, mperr = get_performance_test(i)
    mse.append(mserr)
    mae.append(maerr)
    mpe.append(mperr)
    print('run : ' + str(i) + ', RMSE : ' + str(np.around(np.array(mserr), 2)) + ', MAE : ' + str(np.around(np.array(maerr), 2)) +
          ', MAPE : ' + str(np.around(np.array(mperr), 2)) + ', run time: {:.2f}s'.format(time.time() - t))


mse = np.array(mse)
mae = np.array(mae)
mpe = np.array(mpe)

columns_ = ['SetTransformer', 'STGCN']
df_mse = pd.DataFrame(mse, columns=columns_)
df_mae = pd.DataFrame(mae, columns=columns_)
df_mpe = pd.DataFrame(mpe, columns=columns_)


def get_performance_test_bayesian(trial_i):
    torch.manual_seed(trial_i)
    np.random.seed(trial_i)

    indices = np.arange(0, mean.shape[0])
    np.random.shuffle(indices)
    idx_train = indices[:55]
    idx_test = indices[55:]

    X = train_Set(n=25, seed=trial_i)
    X = np.array(X, dtype='float')

    features = torch.FloatTensor(X)
    labels = torch.FloatTensor(mean)

    mserr = []
    maerr = []
    mperr = []

    models = [
        STGCN(in_features=10)
    ]

    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    for model in models:
        model.apply(weight_reset)

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    def train_(epoch, adj_=None):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if adj_ is not None:
            output, _ = model(features, adj_)
        else:
            output, _ = model(features)
        loss_train = mse(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_test = mse(output[idx_test], labels[idx_test])
        return loss_test.item(), output

    for network in (models):

        t_total = time.time()
        # Model and optimizer
        model = network

        if isinstance(model, (STGCN)):
            embedding = np.loadtxt(log_dir + '/decoding_set_transformer_gcn' + '_trial_' + str(trial_i), delimiter=",")

            adj_np = MAP_inference(embedding, num_neib, 1)
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
            if isinstance(model, SetTransformer):
                value, output = train_(epoch)
            else:
                value, output = train_(epoch, adj_norm)

            if epoch >= EPOCHS - MC_samples:
                output_ += output

        output = output_ / np.float32(MC_samples)

        maerr.append(non_standardized['price'].std() * (mae(output[idx_test], labels[idx_test]).detach().cpu().numpy()))
        mserr.append(
            non_standardized['price'].std() * np.sqrt(mse(output[idx_test], labels[idx_test]).detach().numpy()))

        target = (non_standardized['price'].std() * labels[idx_test]).detach().cpu().numpy() + non_standardized[
            'price'].mean()
        pred = (non_standardized['price'].std() * output[idx_test].T).detach().cpu().numpy() + non_standardized[
            'price'].mean()

        mperr.append(100 * np.mean(np.abs((target - pred) / target)))
        model.apply(weight_reset)
        adj_norm = None
    return mserr, maerr, mperr


mse_b = []
mae_b = []
mpe_b = []
print('B-STGCN')
for i in range(num_trials):
    t = time.time()
    mserr_b, maerr_b, mperr_b = get_performance_test_bayesian(i)
    mse_b.append(mserr_b)
    mae_b.append(maerr_b)
    mpe_b.append(mperr_b)
    print('run : ' + str(i+1) + ', RMSE : ' + str(np.around(np.array(mserr_b), 2)) + ', MAE : ' + str(np.around(np.array(maerr_b), 2)) +
          ', MAPE : ' + str(np.around(np.array(mperr_b), 2)) + ', run time: {:.2f}s'.format(time.time() - t))


mse_b = np.array(mse_b)
mae_b = np.array(mae_b)
mpe_b = np.array(mpe_b)

columns_b = ['B-STGCN']
df_mse_b = pd.DataFrame(mse_b, columns = columns_b)
df_mae_b = pd.DataFrame(mae_b, columns = columns_b)
df_mpe_b = pd.DataFrame(mpe_b, columns = columns_b)

mse_concat = pd.concat([df_mse, df_mse_b], axis=1)
mae_concat = pd.concat([df_mae, df_mae_b], axis=1)
mpe_concat = pd.concat([df_mpe, df_mpe_b], axis=1)

mse_concat.to_csv('rmse_set_transformer.csv', index=False)
mae_concat.to_csv('mae_set_transformer.csv', index=False)
mpe_concat.to_csv('mape_set_transformer.csv', index=False)

