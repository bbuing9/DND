import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.base_dataset import NewsDataset, ReviewDataset, ClincDataset, TRECDataset, SST5Dataset, GLUEDataset, IMDBDataset
from models import load_backbone

from common import CKPT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_base_dataset(data_name, tokenizer, data_ratio=1.0, seed=0):
    print('Initializing base dataset... (name: {})'.format(data_name))

    # Text Classifications
    if data_name == 'news':
        dataset = NewsDataset(tokenizer, data_ratio, seed)
    elif data_name == 'review':
        dataset = ReviewDataset(tokenizer, data_ratio, seed)
    elif data_name == 'trec':
        dataset = TRECDataset(tokenizer, data_ratio, seed)
    elif data_name == 'clinc':
        dataset = ClincDataset(tokenizer, data_ratio, seed)
    elif data_name == 'sst5':
        dataset = SST5Dataset(tokenizer, data_ratio, seed)
    elif data_name == 'imdb':
        dataset = IMDBDataset(tokenizer, data_ratio, seed)
    else:
        if data_name == 'stsb':
            n_class = 1
        elif data_name == 'mnli':
            n_class = 3
        else:
            n_class = 2
        # GLUE TASKs
        dataset = GLUEDataset(data_name, n_class, tokenizer, data_ratio, seed)

    return dataset