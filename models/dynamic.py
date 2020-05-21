import tensorflow as tf
import numpy as np
from scipy.sparse.linalg import eigs

def max_eigs(A):
    b = A.shape[0]
    res = []
    for i in range(b):
        res.append(eigs(A[i,:,:], k=1, which='LR')[0][0].real)
    return np.array(res)

def scaled_laplacian_tf(L, n):
    '''
    Normalized graph Laplacian function. Input W is 
    :param W: tf.tensor, [batch_size, n_route, n_route], weighted adjacency matrix of G.
    :return: tf.tensor, [batch_size, n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    d = tf.linalg.diag_part(L)
    # L -> graph Laplacian
    #L[np.diag_indices_from(L)] = d
    # for i in range(n):
    #     for j in range(n):
    #         if (d[:, i] > 0) and (d[:, j] > 0):
    #             L[:, i, j] = L[:, i, j] / tf.sqrt(d[i] * d[j])
    condition = tf.less(d, 1e-6)
    dp = tf.where(condition, tf.ones_like(d), d)
    D12 = tf.linalg.diag(1. / tf.sqrt(dp))
    Lr = tf.matmul(tf.matmul(D12, L), D12)
    # Lr = L
    # eigv lambda_max \approx 2.0, the largest eigenvalues of L.
    eigv = tf.py_func(max_eigs, [Lr], tf.float32)
    # eigv = tf.linalg.eigh(Lr)[0][:,-1]
    lambda_max = tf.reshape(eigv, [-1, 1, 1])
    return 2 * Lr / lambda_max - tf.eye(n, batch_shape=[1])
    #return L

def cheb_poly_approx_tf(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = tf.zeros_like(L) + tf.eye(n, batch_shape=[1]), L

    if Ks > 1:
        L_list = [L0, L1]
        for i in range(Ks - 2):
            Ln = 2 * L * L1 - L0
            L_list.append(Ln)
            L0, L1 = L1, Ln
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return tf.concat(L_list, axis=-1)
    elif Ks == 1:
        return L0
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def tucker_decomp(x, n_his, n, channel):
    rank_s, rank_t = tf.get_collection("matrix_rank")
    U1 = tf.get_variable(name = "proj_s", shape = [n, rank_s], initializer=tf.random_normal_initializer())
    U2 = tf.get_variable(name = "proj_t", shape = [n_his, rank_t], initializer=tf.random_normal_initializer())
    # Decomposition
    x_tmp_s = tf.reshape(tf.transpose(x, [0, 1, 3, 2]), [-1, n_his * channel, n])
    x_proj_s = tf.reshape(tf.matmul(x_tmp_s, U1), [-1, n_his, channel, rank_s])
    x_tmp_t = tf.transpose(tf.reshape(x_proj_s, [-1, n_his, channel*rank_s]), [0, 2, 1])
    x_proj_t = tf.matmul(x_tmp_t, U2)
    x_restore_t = tf.matmul(x_proj_t, tf.transpose(U2))
    x_tmp_r = tf.reshape(tf.transpose(x_restore_t, [0, 2, 1]), [-1, n_his, channel, rank_s])
    x_tmp_r = tf.reshape(x_tmp_r, [-1, n_his*channel, rank_s])
    x_restore_s = tf.matmul(x_tmp_r, tf.transpose(U1))
    Xs = tf.transpose(tf.reshape(x_restore_s, [-1, n_his, channel, n]), [0, 1, 3, 2])
    Xe = x - Xs
    return Xs, Xe

def unfold_normalization(A):
    _, n_his, n, channel = A.get_shape().as_list()
    A = tf.transpose(A, [0, 1, 3, 2])
    A = tf.reshape(A, [-1, n_his*channel, n])
    A = A - tf.reduce_mean(A, axis=1, keepdims=True)
    A_norm = A / tf.sqrt(float(n))
    return A_norm

def conv_2d(inp):
    w_q1 = tf.get_variable('wq1', shape=[3, 3, 3, 3], dtype=tf.float32)
    w_q2 = tf.get_variable('wq2', shape=[3, 3, 3, 1], dtype=tf.float32)
    b_q1 = tf.get_variable(name='bq1', initializer=tf.zeros([3]), dtype=tf.float32)
    b_q2 = tf.get_variable(name='bq2', initializer=tf.zeros([1]), dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_q1))
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_q2))
    out = tf.nn.conv2d(inp, w_q1, strides=[1, 1, 1, 1], padding='SAME') + b_q1
    out = tf.layers.batch_normalization(out)
    out = tf.nn.relu(out)
    out = tf.nn.conv2d(out, w_q2, strides=[1, 1, 1, 1], padding='SAME') + b_q2
    out = tf.layers.batch_normalization(out)
    return tf.nn.relu(out)

def calc_L_series_sum(B, Ls):
    I = tf.get_collection('series_iteration')[0]
    eta = -1
    factor = tf.matmul(B, Ls)
    # factor = tf.py_func(myPrint, [factor], tf.float32)
    #factor = tf.py_func(calc_max_eigv, [tf.matmul(Ls, B)], tf.float32)
    prod = tf.matmul(Ls, factor)
    Le = eta * prod
    #tf.py_func(myPrint, [Lt], tf.float32)
    for i in range(2, I+1):
        eta *= -1
        prod = tf.matmul(prod, factor)
        Le = Le + eta * prod
    return Ls + Le

def calc_L_inv(n, B, Ls):
    factor = tf.matmul(Ls, B)
    I = tf.eye(n, batch_shape=[1])
    inv = tf.linalg.inv(I + factor)
    Le = tf.matmul(tf.matmul(factor, inv), Ls)
    return Ls - Le

def laplacian_estimator(x, Ks):
    _, n_his, n, channel = x.get_shape().as_list()
    Xs, Xe = tucker_decomp(x, n_his, n, channel)
    # Unfolding Normalization
    Xs_norm, Xe_norm = unfold_normalization(Xs), unfold_normalization(Xe)
    Q_se = tf.matmul(tf.transpose(Xs_norm, [0, 2, 1]), Xe_norm)
    Q_ss = tf.matmul(tf.transpose(Xs_norm, [0, 2, 1]), Xs_norm)
    Q_ee = tf.matmul(tf.transpose(Xe_norm, [0, 2, 1]), Xe_norm)
    # 2D-conv
    Q_input = tf.concat([tf.expand_dims(Q, -1) for Q in [Q_se, Q_ss, Q_ee]], axis=-1)  # [Batch, n, n, 3]
    Ze = conv_2d(Q_input)
    # Estimato B and Le
    B = Q_se + Q_ee + Q_ss + tf.reshape(Ze, [-1, n, n])
    Ls = tf.get_collection("base_graph_kernel")[0]
    Lt = calc_L_series_sum(B, Ls)
    #Lt = calc_L_inv(n, B, Ls)
    #Lt = tf.py_func(myPrint, [Lt], tf.float32)
    # Laplacian Normalization
    Lt = scaled_laplacian_tf(Lt, n)
    pretrain_loss1 = tf.reduce_mean(tf.linalg.trace(tf.matmul(Q_ss, Ls))) / (n * n)
    pretrain_loss2 = tf.sqrt(tf.reduce_mean(Xe_norm ** 2))
    Lt = cheb_poly_approx_tf(Lt, Ks, n)
    return Lt, [pretrain_loss1, pretrain_loss2], None
