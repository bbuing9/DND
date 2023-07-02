import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import datetime
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from eval import test_acc
from data import get_base_dataset
from models import load_backbone, Classifier
from training import train_base, train_mixup, train_aug, train_dnd
from common import CKPT_PATH, parse_args
from utils import Logger, set_seed, set_model_path, save_model, save_policy, print_policy, load_augment, set_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args(mode='train')

    # Set seed
    set_seed(args)

    # Set logs
    log_name = f"{args.dataset}_R{args.data_ratio}_{args.train_type}_B{args.batch_size}_Sim{args.lambda_sim}_Recon{args.lambda_recon}_S{args.seed}"

    logger = Logger(log_name)
    log_dir = logger.logdir

    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    backbone, tokenizer = load_backbone(args.backbone)

    logger.log('Initializing dataset...')
    dataset = get_base_dataset(args.dataset, tokenizer, args.data_ratio, args.seed)
    train_loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(dataset.val_dataset, shuffle=True, batch_size=4, num_workers=4)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

    logger.log('Initializing model and optimizer...')
    model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).to(device)
    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=0)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logger.log('Training model...')
    policy, policy_optimizer = set_policy(args, logger)

    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc = 0, 0

    # Load pre-generated augmented samples
    if 'base' in args.train_type or 'mixup' in args.train_type:
        bts_src, ctx_src, eda_src = None, None, None
    else:
        bts_src, ctx_src, eda_src = load_augment(args)

    for epoch in range(1, args.epochs + 1):
        train_func(args, train_loader, model, optimizer, epoch, logger, bts_src, ctx_src, eda_src, policy, policy_optimizer)
        best_acc, final_acc = eval_func(args, model, val_loader, test_loader, logger, log_dir, dataset, best_acc, final_acc)

    logger.log('===========>>>>> Final Test Accuracy: {}'.format(final_acc))

    # Save policy if our method is trained...
    if 'dnd' in args.train_type:
        save_policy(args, policy, log_dir, dataset)

def train_func(args, train_loader, model, optimizer, epoch, logger, bts_src, ctx_src, eda_src, policy, policy_optimizer):
    if 'base' in args.train_type:
        train_base(args, train_loader, model, optimizer, epoch, logger)
    elif 'mixup' in args.train_type:
        train_mixup(args, train_loader, model, optimizer, epoch, logger)
    elif 'aug' in args.train_type:
        train_aug(args, train_loader, model, optimizer, bts_src, ctx_src, eda_src, epoch, logger, policy)
    elif 'dnd' in args.train_type:
        if epoch % 2 == 0:
            print_policy(policy, logger)
        train_dnd(args, train_loader, model, optimizer, bts_src, ctx_src, eda_src, policy, policy_optimizer, epoch, logger)
    else:
        print("========== Please set the proper training method ==========")

def eval_func(args, model, val_loader, test_loader, logger, log_dir, dataset, best_acc, final_acc):
    # other_metric; [mcc, f1, p, s]
    acc, other_metric = test_acc(args, val_loader, model, logger)

    if args.dataset == 'cola':
        metric = other_metric[0]
    elif args.dataset == 'stsb':
        metric = other_metric[2]
    else:
        metric = acc

    if metric >= best_acc:
        # As val_data == test_data in GLUE, do not inference it again.
        if args.dataset == 'wnli' or args.dataset == 'rte' or args.dataset == 'mrpc' or args.dataset == 'stsb' or \
                args.dataset == 'cola' or args.dataset == 'sst2' or args.dataset == 'qnli' or args.dataset == 'qqp':
            t_acc, t_other_metric = acc, other_metric
        else:
            t_acc, t_other_metric = test_acc(args, test_loader, model, logger)

        if args.dataset == 'cola':
            t_metric = t_other_metric[0]
        elif args.dataset == 'stsb':
            t_metric = t_other_metric[2]
        else:
            t_metric = t_acc

        # Update test accuracy based on validation performance
        best_acc = metric
        final_acc = t_metric

        if args.dataset == 'mrpc' or args.dataset == 'qqp':
            logger.log('========== Test Acc/F1 ==========')
            logger.log('Test acc: {:.3f} Test F1: {:.3f}'.format(final_acc, t_other_metric[1]))
        elif args.dataset == 'stsb':
            logger.log('========== Test P/S ==========')
            logger.log('Test P: {:.3f} Test S: {:.3f}'.format(t_other_metric[2], t_other_metric[3]))
        elif args.dataset == 'mnli':
            logger.log('========== Test m/mm ==========')
            logger.log('Test matched/mismatched: {:.3f}/{:.3f}'.format(best_acc, final_acc))
        else:
            logger.log('========== Val Acc ==========')
            logger.log('Val acc: {:.3f}'.format(best_acc))
            logger.log('========== Test Acc ==========')
            logger.log('Test acc: {:.3f}'.format(final_acc))

        # Save model
        if args.save_ckpt:
            logger.log('Save model...')
            save_model(args, model, log_dir, dataset)

    return best_acc, final_acc

if __name__ == "__main__":
    main()
