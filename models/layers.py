# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import tensorflow as tf
from utils.math_graph import *


def gconv(x, Lt, T, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, t, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [batch_size, n_route, Ks*n_route]
    # kernel = tf.get_collection('graph_kernel')[0]
    kernel = Lt
    n = tf.shape(kernel)[1]
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 1, 3, 2]), [-1, T*c_in, n]) #[b, t, n, c] -> [b, t*c, n]
    #x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.reshape(tf.matmul(x_tmp, kernel), [-1, T, c_in, Ks, n]), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]

    if act_func == 'GLU':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    else:
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def spatio_conv_layer(x, Lt, Ks, c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
    #x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), Lt, ws, Ks, c_in, c_out) + bs
    x_gconv = gconv(x, Lt, T, ws, Ks, c_in, c_out) + bs
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


def st_conv_block(x, Lt, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param scope: str, variable scope.
    :param keep_prob: placeholder, prob of dropout.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    c_si, c_t, c_oo = channels

    with tf.variable_scope(f'stn_block_{scope}_in'):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
        x_t = spatio_conv_layer(x_s, Lt, Ks, c_t, c_t)
    with tf.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, act_func='GLU'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)

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

def conv_2d(inp):
    w_q1 = tf.get_variable('wq_input1', shape=[3, 3, 3, 3], dtype=tf.float32)
    w_q2 = tf.get_variable('wq_input2', shape=[3, 3, 3, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_q1))
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_q2))
    out = tf.nn.conv2d(inp, w_q1, strides=[1, 1, 1, 1], padding='SAME')
    out = tf.layers.batch_normalization(out)
    out = tf.nn.relu(out)
    out =  tf.nn.conv2d(out, w_q2, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(out)

def approx_L(B, Ls):
    I = tf.get_collection('series_iteration')[0]
    eta = -1
    factor = tf.matmul(B, Ls)
    # factor = tf.py_func(myPrint, [factor], tf.float32)
    #factor = tf.py_func(calc_max_eigv, [tf.matmul(Ls, B)], tf.float32)
    prod = tf.matmul(Ls, factor)
    Lt = Ls + eta * prod
    #tf.py_func(myPrint, [Lt], tf.float32)
    for i in range(2, I+1):
        eta *= -1
        prod = tf.matmul(prod, factor)
        Lt = Lt + eta * prod
    return Lt

def calc_L(n, B, Ls):
    factor = tf.matmul(Ls, B)
    I = tf.eye(n, batch_shape=[1])
    inv = tf.linalg.inv(I + factor)
    Le = tf.matmul(tf.matmul(factor, inv), Ls)
    return Ls + Le

def laplacian_estimator(x, Ks):
    _, n_his, n, channel = x.get_shape().as_list()
    Xs, Xe = tucker_decomp(x, n_his, n, channel)
    # Unfolding Normalization
    Xs = tf.transpose(Xs, [0, 1, 3, 2])
    Xe = tf.transpose(Xe, [0, 1, 3, 2])
    Xs = tf.reshape(Xs, [-1, n_his*channel, n])
    Xs = Xs - tf.reduce_mean(Xs, axis=1, keepdims=True)
    Xe = tf.reshape(Xe, [-1, n_his*channel, n])
    Xe = Xe - tf.reduce_mean(Xe, axis=1, keepdims=True)
    Xs_norm = Xs / tf.sqrt(tf.reduce_sum(Xs ** 2, axis=1, keep_dims=True) + 1e-12)
    Xe_norm = Xe / tf.sqrt(tf.reduce_sum(Xe ** 2, axis=1, keep_dims=True) + 1e-12)
    Q_se = tf.matmul(tf.transpose(Xs_norm, [0, 2, 1]), Xe_norm)
    Q_ss = tf.matmul(tf.transpose(Xs_norm, [0, 2, 1]), Xs_norm)
    Q_ee = tf.matmul(tf.transpose(Xe_norm, [0, 2, 1]), Xe_norm)
    # 2D-conv
    Q_input = tf.concat([tf.expand_dims(Q, -1) for Q in [Q_se, Q_ss, Q_ee]], axis=-1)  # [Batch, n, n, 3]
    Ze = conv_2d(Q_input)
    # Estimato B and Le
    B = Q_se + Q_ee + Q_ss + tf.reshape(Ze, [-1, n, n])
    Ls = tf.get_collection("base_graph_kernel")[0]
    # Lt = apprx_L(B, Ls)
    Lt = calc_L(n, B, Ls)
    #Lt = tf.py_func(myPrint, [Lt], tf.float32)
    # Laplacian Normalization
    Lt = scaled_laplacian_tf(Lt, n)
    pretrain_loss1 = tf.reduce_mean(tf.linalg.trace(tf.matmul(Q_ss, Ls))) / (n * n)
    pretrain_loss2 = tf.sqrt(tf.reduce_mean(Xe_norm ** 2))
    Lt = cheb_poly_approx_tf(Lt, Ks, n)
    return Lt, [pretrain_loss1, pretrain_loss2], None

# def trace(a, b):
#     return tf.reduce_sum(tf.linalg.diag_part(tf.matmul(a, b)), axis=-1)