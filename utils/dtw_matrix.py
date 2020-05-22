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
            M[i, j] = np.sum(np.abs(X[i,:] - Y[j,:]), axis=-1)
            if i == 0 and j == 0:
                plus = 0
            elif i == 0:
                plus = M[i, j-1]
            elif j == 0:
                plus = M[i-1, j]
            elif j == i-T:
                plus = min(M[i-1, j-1], M[i-1, j])
            elif j == i+T:
                plus = min(M[i-1, j-1], M[i, j-1])
            else:
                plus = min(M[i-1, j-1], M[i, j-1], M[i-1, j])
            M_c[i, j] = M[i, j] ** 2 + plus
    return np.sqrt(M_c[nt-1, nt-1])

@jit
def dtw_weight_matrix(X, n, args):
    W = np.zeros((n, n))
    for i in range(n):
        list_dist = []
        for j in range(n):
            list_dist.append(dtw_distance(X[i], X[j], args.time_interval))
        top_arg = np.argsort(list_dist)
        for k in range(args.topk):
            jj = top_arg[k]
            W[i, jj] = W[jj, i] = 1
        if i % 10 == 0:
            print(i)
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
