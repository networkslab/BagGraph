import numpy as np
from numpy import matlib
from scipy.stats import t
import scipy.sparse as sp
from math import sqrt
from statistics import stdev


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv_sqrt = np.sqrt(r_inv)
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    mx = r_mat_inv_sqrt.dot(mx)
    mx = mx.dot(r_mat_inv_sqrt)
    return mx


def accuracy(labels, output):
    preds = (output > 0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def compute_distance(embed):
    N = embed.shape[0]
    p = np.dot(embed, np.transpose(embed))
    q = np.matlib.repmat(np.diag(p), N, 1)
    dist = q + np.transpose(q) - 2 * p
    dist[dist < 1e-8] = 1e-8
    return dist


def estimate_graph(gamma, epsilon, dist, max_iter, k, r):
    np.random.seed(0)

    N = dist.shape[0]
    dist += 1e10 * np.eye(N)

    deg_exp = np.minimum(int(N-1), int(k * r))

    dist_sort_col_idx = np.argsort(dist, axis=0)
    dist_sort_col_idx = np.transpose(dist_sort_col_idx[0:deg_exp, :])

    dist_sort_row_idx = np.matlib.repmat(np.arange(N).reshape(N, 1), 1, deg_exp)

    dist_sort_col_idx = np.reshape(dist_sort_col_idx, int(N * deg_exp)).astype(int)
    dist_sort_row_idx = np.reshape(dist_sort_row_idx, int(N * deg_exp)).astype(int)

    dist_idx = np.zeros((int(N * deg_exp), 2)).astype(int)
    dist_idx[:, 0] = dist_sort_col_idx
    dist_idx[:, 1] = dist_sort_row_idx
    dist_idx = np.sort(dist_idx, axis=1)
    dist_idx = np.unique(dist_idx, axis=0)
    dist_sort_col_idx = dist_idx[:, 0]
    dist_sort_row_idx = dist_idx[:, 1]

    num_edges = len(dist_sort_col_idx)

    w_init = np.random.uniform(0, 1, size=(num_edges, 1))
    d_init = k * np.random.uniform(0, 1, size=(N, 1))

    w_current = w_init
    d_current = d_init

    dist_sorted = np.sort(dist, axis=0)

    B_k = np.sum(dist_sorted[0:k, :], axis=0)
    dist_sorted_k = dist_sorted[k-1, :]
    dist_sorted_k_plus_1 = dist_sorted[k, :]

    theta_lb = 1 / np.sqrt(k * dist_sorted_k_plus_1 ** 2 - B_k * dist_sorted_k_plus_1)
    theta_lb = theta_lb[~np.isnan(theta_lb)]
    theta_lb = theta_lb[~np.isinf(theta_lb)]
    theta_lb = np.mean(theta_lb)

    theta_ub = 1 / np.sqrt(k * dist_sorted_k ** 2 - B_k * dist_sorted_k)
    theta_ub = theta_ub[~np.isnan(theta_ub)]
    theta_ub = theta_ub[~np.isinf(theta_ub)]
    theta_ub = np.mean(theta_ub)

    theta = (theta_lb + theta_ub) / 2

    dist = theta * dist

    z = dist[dist_sort_row_idx, dist_sort_col_idx]
    z.shape = (num_edges, 1)

    for iter in range(max_iter):

        # print('Graph inference epoch : ' + str(iter))

        St_times_d = d_current[dist_sort_row_idx] + d_current[dist_sort_col_idx]
        y_current = w_current - gamma * (2 * w_current + St_times_d)

        adj_current = np.zeros((N, N))
        adj_current[dist_sort_row_idx, dist_sort_col_idx] = np.squeeze(w_current)
        adj_current = adj_current + np.transpose(adj_current)
        S_times_w = np.sum(adj_current, axis=1)
        S_times_w.shape = (N, 1)
        y_bar_current = d_current + gamma * S_times_w

        p_current = np.maximum(0, np.abs(y_current) - 2 * gamma * z)
        p_bar_current = (y_bar_current - np.sqrt(y_bar_current * y_bar_current + 4 * gamma)) / 2

        St_times_p_bar = p_bar_current[dist_sort_row_idx] + p_bar_current[dist_sort_col_idx]
        q_current = p_current - gamma * (2 * p_current + St_times_p_bar)

        p_matrix_current = np.zeros((N, N))
        p_matrix_current[dist_sort_row_idx, dist_sort_col_idx] = np.squeeze(p_current)
        p_matrix_current = p_matrix_current + np.transpose(p_matrix_current)
        S_times_p = np.sum(p_matrix_current, axis=1)
        S_times_p.shape = (N, 1)
        q_bar_current = p_bar_current + gamma * S_times_p

        w_updated = np.abs(w_current - y_current + q_current)
        d_updated = np.abs(d_current - y_bar_current + q_bar_current)

        if (np.linalg.norm(w_updated - w_current) / np.linalg.norm(w_current) < epsilon) and \
                (np.linalg.norm(d_updated - d_current) / np.linalg.norm(d_current) < epsilon):
            break
        else:
            w_current = w_updated
            d_current = d_updated

    upper_tri_index = np.triu_indices(N, k=1)

    z = dist[upper_tri_index[0], upper_tri_index[1]]
    z.shape = (int(N * (N - 1) / 2), 1)
    z = z * np.max(w_current)

    w_current = w_current / np.max(w_current)

    inferred_graph = np.zeros((N, N))
    inferred_graph[dist_sort_row_idx, dist_sort_col_idx] = np.squeeze(w_current)
    inferred_graph = inferred_graph + np.transpose(inferred_graph) + np.eye(N)

    return inferred_graph


def MAP_inference(x, num_neib, r):
    N = x.shape[0]
    k = int(1 * num_neib)

    dist = compute_distance(x)

    inferred_graph = estimate_graph(0.01, 0.001, dist, 1000, k, r)

    return inferred_graph
