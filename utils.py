import os
import sys
import time
from datetime import datetime
import shutil
import math

import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from autoaug.policy import Policy_gumbel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        logdir = 'logs/' + fn
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if len(os.listdir(logdir)) != 0:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def set_model_path(args, dataset):
    # Naming the saving model
    suffix = "_"
    suffix += str(args.train_type)

    return dataset.base_path + suffix + '.model'

def save_model(args, model, log_dir, dataset):
    # Save the model
    if isinstance(model, nn.DataParallel):
        model = model.module

    os.makedirs(log_dir, exist_ok=True)
    model_path = set_model_path(args, dataset)
    save_path = os.path.join(log_dir, model_path)
    torch.save(model.state_dict(), save_path)

def save_policy(args, policy, log_dir, dataset):
    os.makedirs(log_dir, exist_ok=True)
    policy_path = set_model_path(args, dataset)
    save_path = os.path.join(log_dir, policy_path + '_policy')
    torch.save(policy.state_dict(), save_path)

def print_policy(policy, logger):
    logger.log(policy)
    for i in range(4):
        logger.log("=== {} ===".format(i))
        logger.log(np.round(1000 * policy.sub_policies[i].stages[0].weights.data.cpu().numpy()) / 1000)
        logger.log(np.round(1000 * policy.sub_policies[i].stages[1].weights.data.cpu().numpy()) / 1000)

def load_augment(args):
    bts_src = np.load('./pre_augment/' + args.dataset + '_' + str(args.data_ratio) + '_backtrans_' + args.backbone +'.npy')[:, 0, :]
    ctx_src = np.load('./pre_augment/' + args.dataset + '_' + str(args.data_ratio) + '_contextual_' + args.backbone +'.npy')[:, 0, :]
    eda_src = np.load('./pre_augment/' + args.dataset + '_' + str(args.data_ratio) + '_eda_'+ args.backbone +'.npy')[:, 0, :]

    return torch.LongTensor(bts_src), torch.LongTensor(ctx_src), torch.LongTensor(eda_src)

def set_policy(args, logger):
    if 'dnd' in args.train_type  or args.policy:
        policy = Policy_gumbel.faster_auto_augment_policy(num_sub_policies=4, temperature=args.policy_temp, operation_count=2, num_chunks=1)
        policy_optimizer = optim.Adam(policy.parameters(), lr=args.policy_lr)

        if args.pre_policy is not None:
            policy_weight = torch.load(args.pre_policy, map_location='cpu')
            policy.load_state_dict(policy_weight)

            logger.log('Load a pre-trained augmentation policy...')
            print_policy(policy, logger)
    else:
        policy = None
        policy_optimizer = None

    return policy, policy_optimizer