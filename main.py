# main.py

# SPDX-License-Identifier: MIT
# See COPYING file for more details.

import os
import torch
import argparse
from torch.backends import cudnn
from model import build_net
from train import _train
from eval import _eval
import numpy as np
import random

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Data
    parser.add_argument('--dataset', type=str, default='./Haze4K')

    # Train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='results/model.pkl')

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()

    args.model_save_dir = os.path.join('results/')
    args.result_dir = os.path.join('results/','test')

    args.data_dir = args.dataset
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    main(args)
