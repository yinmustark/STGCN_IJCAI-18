# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import warnings
warnings.filterwarnings("ignore")

import argparse

config = tf.ConfigProto() 
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--rank', type=int, nargs=2, default=[10,3])
parser.add_argument('--I', type=int, default=6)
parser.add_argument('-pe', '--pretrain_epoch', type=int, default=31)
parser.add_argument('-p', '--save_path', type=str, default='./output/')
parser.add_argument('-t', '--test', action='store_true')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]
tf.add_to_collection(name='pretrain_beta', value=args.beta)
tf.add_to_collection(name='matrix_rank', value=int(args.rank[0]))
tf.add_to_collection(name='matrix_rank', value=int(args.rank[1]))
tf.add_to_collection(name='series_iteration', value=args.I)
sum_path = pjoin(args.save_path, 'tensorboard/')

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./dataset', f'PeMSD7_W_{n}.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))

# Calculate graph kernel
Ls = scaled_laplacian(W)[None, :, :]
# Alternative approximation method: 1st approx - first_approx(W, n).
tf.add_to_collection(name='base_graph_kernel', value=tf.cast(tf.constant(Ls), tf.float32))

# Data Preprocessing
data_file = f'PeMSD7_V_{n}.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    if args.test:
        model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode, load_path=pjoin(args.save_path, 'models'))
    else:
        model_train(PeMS, blocks, args, sum_path)
        model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode, load_path=pjoin(args.save_path, 'models'))
