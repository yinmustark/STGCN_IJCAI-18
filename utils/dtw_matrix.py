import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from os.path import exists as pexists, join as pjoin
import argparse
from numba import jit

@jit
def dtw_distance(X, Y, T):
    nt = X.shape[0]
    M = np.zeros((nt, nt))
    M_c = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(max(i-T, 0), min(i+T+1, nt)):
            tmp = np.sum((X[i,:] - Y[j,:]) ** 2)
            M[i, j] = np.sqrt(tmp)
            if i == 0 and j == 0:
                plus = 0
            elif i == 0:
                plus = M_c[i, j-1]
            elif j == 0:
                plus = M_c[i-1, j]
            elif j == i-T:
                plus = min(M_c[i-1, j-1], M_c[i-1, j])
            elif j == i+T:
                plus = min(M_c[i-1, j-1], M_c[i, j-1])
            else:
                plus = min(M_c[i-1, j-1], M_c[i, j-1], M_c[i-1, j])
            M_c[i, j] = tmp + plus
    return np.sqrt(M_c[nt-1, nt-1])

@jit
def dtw_weight_matrix(X, n, args):
    W = np.zeros((n, n))
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and not dist[i, j]:
                dist[i, j] = dtw_distance(X[i], X[j], args.time_interval)
                dist[j, i] = dist[i, j]
        if(i%50 == 0):
            print(i)
    dist_sort_arg = np.argsort(dist, axis=1)
    for i in range(n):
        j_indices = dist_sort_arg[i, :args.topk]
        W[i, j_indices] = W[j_indices, i] = 1
    return W

def dtw_adj_matrix(args, n_train):
    weight_path = pjoin('./dataset/graph/', f'dtw_adj_T{args.time_interval}_k{args.topk}.npy')
    if pexists(weight_path):
        return np.load(weight_path)
    
    n = args.n_route
    nt = 288
    data_path = pjoin('./dataset', f'PeMSD7_V_{n}.csv')
    data = pd.read_csv(data_path, header=None).values
    data_p = np.concatenate([data, np.zeros((1, n))])
    X = data.reshape([-1, nt, n])[:n_train]
    X = X.transpose([2, 1, 0])

    W = dtw_weight_matrix(X, n, args)

    np.save(weight_path, W)
    return W

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_route', type=int, default=228)
    parser.add_argument('-T', '--time_interval', type=int, default=12)
    parser.add_argument('-k', '--topk', type=int, default=8)
    args = parser.parse_args()

    print(dtw_adj_matrix(args, 34).sum() / (228*228))
