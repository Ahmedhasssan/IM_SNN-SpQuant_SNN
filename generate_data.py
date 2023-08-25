import os
import torch
import numpy as np
import random
import argparse
import logging
from dvsloader.dvs2dataset import *

parser = argparse.ArgumentParser(description='PyTorch DVS Data Generator')
parser.add_argument('--dataset', type=str, default='dvscifar10,ncars,ncaltech', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--root_dir', type=str, default='"/home2/ahasssan/data/dvscifar10/', help='data directory')
parser.add_argument('--save_dir', type=str, default='./data/', help='save data directory')
parser.add_argument('--T', default=10, type=int, metavar='N', help='snn simulation time (default: 2)')

args = parser.parse_args()

def dvs_data(args):
    root = args.root_dir
    save_dir = args.save_dir
    T = args.T
    dataset = args.dataset
    height = 128
    width = 128
    mode=""
    loader = DVSLoader(root, mode, height=height, width=width, save_dir=save_dir)
    loader.event2quene(nframes=T, save=True, get_hist=False, dataset=dataset)

if __name__ == "__main__":
    dvs_data(args)