import os
import json
from abc import *

import torch
import csv
from torch.utils.data import TensorDataset
import numpy as np

from common import DATA_PATH

def create_tensor_dataset(inputs, labels, index):
    assert len(inputs) == len(labels)
    assert len(inputs) == len(index)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels).unsqueeze(1)  # (N, 1)
    index = np.array(index)
    index = torch.Tensor(index).long()

    dataset = TensorDataset(inputs, labels, index)

    return dataset

class BaseDataset(metaclass=ABCMeta):
    def __init__(self, data_name, total_class, tokenizer, data_ratio=1.0, seed=0):

        self.data_name = data_name
        self.total_class = total_class
        self.root_dir = os.path.join(DATA_PATH, data_name)

        self.tokenizer = tokenizer
        self.data_ratio = data_ratio
        self.seed = seed

        self.n_classes = int(self.total_class)  # Split a given data
        self.class_idx = list(range(self.n_classes))  # all classes
        self.max_class = 1000

        if self.data_ratio < 1.0:
            n_samples = [int(self.max_class * self.data_ratio)] * self.n_classes
            self.n_samples = n_samples
        else:
            self.n_samples = [100000] * self.n_classes

        if not self._check_exists():
            self._preprocess()

        self.train_dataset = torch.load(self._train_path)
        self.val_dataset = torch.load(self._val_path)
        self.test_dataset = torch.load(self._test_path)

    @property
    def base_path(self):
        if self.data_ratio < 1.0:
            base_path = '{}_{}_data_{:.3f}'.format(
                self.data_name, self.tokenizer.name, self.data_ratio, self.seed)
        else:
            base_path = '{}_{}'.format(self.data_name, self.tokenizer.name)

        return base_path

    @property
    def _train_path(self):
        return os.path.join(self.root_dir, self.base_path + '_train.pth')

    @property
    def _val_path(self):
        return os.path.join(self.root_dir, self.base_path + '_val.pth')

    @property
    def _test_path(self):
        return os.path.join(self.root_dir, self.base_path + '_test.pth')

    def _check_exists(self):
        if not os.path.exists(self._train_path):
            return False
        elif not os.path.exists(self._val_path):
            return False
        elif not os.path.exists(self._test_path):
            return False
        else:
            return True

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def _load_dataset(self, *args, **kwargs):
        pass

class NewsDataset(BaseDataset):
    def __init__(self, tokenizer, data_ratio=1.0, seed=0):
        super(NewsDataset, self).__init__('news', 20, tokenizer, data_ratio, seed)

    def _preprocess(self):
        print('Pre-processing news dataset...')
        train_dataset, val_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'test']

        if mode == 'test':
            source_path = os.path.join(self.root_dir, 'test.txt')
        else:
            source_path = os.path.join(self.root_dir, 'train.txt')

        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs, labels, index = [], [], []
        v_inputs, v_labels, v_index = [], [], []

        # Dummy for selecting all samples
        n_samples_train = [100000] * self.n_classes
        n_samples_val = [100000] * self.n_classes
        n_samples_test = [100000] * self.n_classes

        # Count the number of training examples to construct validation set
        n_samples_train_np = np.zeros(self.n_classes)
        if self.data_ratio == 1:
            for line in lines:
                toks = line.split(',')

                if not int(toks[1]) in self.class_idx:  # only selected classes
                    continue

                label = self.class_idx.index(int(toks[1]))  # convert to subclass index
                n_samples_train_np[label] += 1
                n_samples_val = list(np.round(0.1 * n_samples_train_np))

        num, num_v = 0, 0
        for line in lines:
            toks = line.split(',')

            if not int(toks[1]) in self.class_idx:  # only selected classes
                continue

            path = os.path.join(self.root_dir, '{}'.format(toks[0]))
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()

            if not raw_text:
                text = self.tokenizer.encode(text, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                             return_tensors='pt')[0]

            label = self.class_idx.index(int(toks[1]))  # convert to subclass index
            label = torch.tensor(label).long()

            if mode == 'train':
                if n_samples_val[int(label)] > 0:
                    v_inputs.append(text)
                    v_labels.append(label)
                    v_index.append(num_v)

                    n_samples_val[int(label)] -= 1
                    num_v += 1
                elif n_samples_train[int(label)] > 0:
                    inputs.append(text)
                    labels.append(label)
                    index.append(num)

                    n_samples_train[int(label)] -= 1
                    num += 1

            else:
                inputs.append(text)
                labels.append(label)
                index.append(num)

                n_samples_test[int(label)] -= 1
                num += 1

        if raw_text:
            dataset = zip(inputs, labels, index)
        else:
            dataset = create_tensor_dataset(inputs, labels, index)

        if mode == 'train':
            v_dataset = create_tensor_dataset(v_inputs, v_labels, v_index)
            return dataset, v_dataset
        else:
            return dataset

class ReviewDataset(BaseDataset):
    def __init__(self, tokenizer, data_ratio=1.0, seed=0):
        self.train_test_ratio = 0.3  # split ratio for train/test dataset
        super(ReviewDataset, self).__init__('review', 50, tokenizer, data_ratio, seed)

    def _preprocess(self):
        print('Pre-processing review dataset...')
        source_path = os.path.join(self.root_dir, '50EleReviews.json')
        with open(source_path, encoding='utf-8') as f:
            docs = json.load(f)

        np.random.seed(self.seed)  # fix random seed

        train_inds = []
        val_inds = []
        test_inds = []

        max_class = self.max_class  # samples are ordered by class (previous: per_class)
        if self.data_ratio == 1:
            num_test = int(max_class * self.train_test_ratio)
            per_class = int(0.9 * max_class * (1 - self.train_test_ratio))
            per_val = int(0.1 * max_class * (1 - self.train_test_ratio))
        else:
            num_test = int(max_class * self.train_test_ratio)
            per_class = int(0.9 * max_class * self.data_ratio)
            per_val = max(int(0.1 * max_class * self.data_ratio), 1)

        for cls in self.class_idx:  # only selected classes
            shuffled = np.arange(max_class)

            # 0 ~ 630: train, 630 ~ 700: val, 700 ~ 1000: test split
            train_inds += (cls * max_class + shuffled[:per_class]).tolist()
            val_inds += (cls * max_class + shuffled[630:630 + per_val]).tolist()
            test_inds += (cls * max_class + shuffled[-num_test:]).tolist()

        train_dataset = self._load_dataset(docs, train_inds, 'train')
        val_dataset = self._load_dataset(docs, val_inds, 'val')
        test_dataset = self._load_dataset(docs, test_inds, 'test')

        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, docs, indices, mode='train', raw_text=False):
        assert mode in ['train', 'val', 'test']

        inputs = []
        labels = []
        indexs = []

        n_samples = np.zeros(self.n_classes)
        for i in indices:
            if raw_text:
                text = docs['X'][i]
            else:
                text = self.tokenizer.encode(docs['X'][i], add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                             return_tensors='pt')[0]

            label = self.class_idx.index(int(docs['y'][i]))  # convert to subclass index
            label = torch.tensor(label).long()

            inputs.append(text)
            labels.append(label)
            indexs.append(i)

            n_samples[int(label)] += 1

        if raw_text:
            dataset = zip(inputs, labels, indexs)
        else:
            dataset = create_tensor_dataset(inputs, labels, indexs)

        return dataset

class IMDBDataset(BaseDataset):
    def __init__(self, tokenizer, data_ratio=1.0, seed=0):
        self.class_dict = {'pos': 1, 'neg': 0}
        super(IMDBDataset, self).__init__('imdb', 2, tokenizer, data_ratio, seed)

    def _preprocess(self):
        print('Pre-processing imdb dataset...')
        train_dataset, val_dataset, test_dataset = self._load_dataset('both')
        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='both', raw_text=False):
        assert mode in ['both']

        source_path = os.path.join(self.root_dir, 'imdb.txt')
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        train_inputs, train_labels, train_indexs = [], [], []
        val_inputs, val_labels, val_indexs = [], [], []
        test_inputs, test_labels, test_indexs = [], [], []

        n_sample_train = 100000  # Non-meaning large number for selecting all samples
        n_sample_val = 1250
        n_sample_test = 100000

        n_samples_train = [n_sample_train] * self.n_classes
        n_samples_val = [n_sample_val] * self.n_classes
        n_samples_test = [n_sample_test] * self.n_classes

        temp = 0
        for line in lines:
            toks = line.split('\t')

            if len(toks) > 5:  # text contains tab
                text = '\t'.join(toks[2:-2])
                toks = toks[:2] + [text] + toks[-2:]

            if raw_text:
                text = toks[2]
            else:
                text = self.tokenizer.encode(toks[2], add_special_tokens=True, max_length=256, pad_to_max_length=True,
                                             return_tensors='pt')[0]

            if toks[3] == 'unsup':
                continue
            else:
                label = self.class_dict[toks[3]]  # convert to class index
                label = torch.tensor(label).long()

            if toks[1] == 'train':
                if n_samples_val[int(label)] > 0:
                    val_inputs.append(text)
                    val_labels.append(label)
                    val_indexs.append(temp)

                    n_samples_val[int(label)] -= 1
                elif n_samples_train[int(label)] > 0:
                    train_inputs.append(text)
                    train_labels.append(label)
                    train_indexs.append(temp)

                    n_samples_train[int(label)] -= 1
            else:
                if n_samples_test[int(label)] > 0:
                    test_inputs.append(text)
                    test_labels.append(label)
                    test_indexs.append(temp)

                    n_samples_test[int(label)] -= 1
            temp += 1

        if raw_text:
            train_dataset = zip(train_inputs, train_labels)
            val_dataset = zip(val_inputs, val_labels)
            test_dataset = zip(test_inputs, test_labels)
        else:
            train_dataset = create_tensor_dataset(train_inputs, train_labels, train_indexs)
            val_dataset = create_tensor_dataset(val_inputs, val_labels, val_indexs)
            test_dataset = create_tensor_dataset(test_inputs, test_labels, test_indexs)

        return train_dataset, val_dataset, test_dataset

class SST5Dataset(BaseDataset):
    '''
    Source:
    '''
    def __init__(self, tokenizer, data_ratio=1.0, seed=0):
        super(SST5Dataset, self).__init__('sst5', 5, tokenizer, data_ratio, seed)

    def _preprocess(self):
        print('Pre-processing sst5 dataset...')
        train_dataset = self._load_dataset('train')
        val_dataset = self._load_dataset('dev')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'dev', 'test']

        n_samples = list(self.n_samples)
        v_samples = [max(1, int(0.1 * n_samples[0]))] * self.n_classes

        loc_data = "./dataset/sst5/sst5_raw_" + mode + "_data.txt"
        loc_label = "./dataset/sst5/sst5_raw_" + mode + "_label.txt"

        with open(loc_data, "r") as fp:
            raw_data = json.load(fp)

        with open(loc_label, "r") as fp:
            raw_label = json.load(fp)

        inputs, labels, indices = [], [], []

        n_train = np.zeros(5)
        for (i, text) in enumerate(raw_data):
            if raw_text:
                text = text
            else:
                text = self.tokenizer.encode(text, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                             return_tensors='pt')[0]

            label = raw_label[i]
            label = torch.tensor(label).long()

            if mode == 'train':
                if n_samples[int(label)] > 0:
                    inputs.append(text)
                    labels.append(label)
                    indices.append(i)

                    n_samples[int(label)] -= 1
            elif mode == 'dev':
                if v_samples[int(label)] > 0:
                    inputs.append(text)
                    labels.append(label)
                    indices.append(i)

                    v_samples[int(label)] -= 1
            else:
                inputs.append(text)
                labels.append(label)
                indices.append(i)

            n_train[int(label)] += 1

        if raw_text:
            dataset = zip(inputs, labels, indices)
        else:
            dataset = create_tensor_dataset(inputs, labels, indices)

        return dataset


class ClincDataset(BaseDataset):
    def __init__(self, tokenizer, data_ratio=1.0, seed=0):
        super(ClincDataset, self).__init__('clinc', 150, tokenizer, data_ratio, seed)

    def _preprocess(self):
        print('Pre-processing review dataset...')
        source_path = os.path.join(self.root_dir, 'data_small.json')
        with open(source_path, encoding='utf-8') as f:
            docs = json.load(f)

        train_dataset = self._load_dataset(docs, 'train')
        val_dataset = self._load_dataset(docs, 'val')
        test_dataset = self._load_dataset(docs, 'test')

        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def get_label_list(self, docs):
        # Generate label list
        label_list = []
        for i in range(len(docs['train'])):
            labels = docs['train'][i][1]

            if labels not in label_list:
                label_list.append(labels)

        # Generate dictionary for label mapping
        label_mapping = {}
        temp = 0
        for label in label_list:
            label_mapping[label] = temp
            temp += 1

        return label_mapping

    def _load_dataset(self, docs, mode='train', raw_text=False):
        assert mode in ['train', 'val', 'test']

        inputs = []
        labels = []
        indexs = []

        data = docs[mode]
        labeler = self.get_label_list(docs)
        num_samples = list(self.n_samples)

        for i in range(len(data)):
            if raw_text:
                text = data[i][0]
            else:
                text = self.tokenizer.encode(data[i][0], add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                             return_tensors='pt')[0]

            label = labeler[data[i][1]]
            label = torch.tensor(label).long()

            if mode == 'train' and self.data_ratio < 1:
                if num_samples[int(label.data)] > 0:
                    inputs.append(text)
                    labels.append(label)
                    indexs.append(i)

                    num_samples[int(label.data)] -= 1
            else:
                inputs.append(text)
                labels.append(label)
                indexs.append(i)

        if raw_text:
            dataset = zip(inputs, labels, indexs)
        else:
            dataset = create_tensor_dataset(inputs, labels, indexs)

        return dataset

class TRECDataset(BaseDataset):
    def __init__(self, tokenizer, data_ratio=1.0, seed=0):
        super(TRECDataset, self).__init__('trec', 6, tokenizer, data_ratio, seed)

    def _preprocess(self):
        print('Pre-processing trec dataset...')
        train_dataset, val_dataset = self._load_dataset(mode='train')
        test_dataset = self._load_dataset(mode='dev')

        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'dev']

        source_path = os.path.join('./dataset/trec', '{}.txt'.format(mode))
        label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
        num_class = len(label_dict)

        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

            # Check the number of labels in dataset
            num_labels = np.zeros(num_class)
            for line in lines:
                bef_labels = line.split(':')[0]
                bef_input = line[len(bef_labels) + 1:-1]

                fine_label = bef_input.split(' ')[0]
                aft_label = label_dict[bef_labels]

                num_labels[aft_label] += 1

            num_val_full = list(np.round(0.1 * num_labels + 1e-6))

            # Construction of dataset
            inputs, labels, indices = [], [], []
            v_inputs, v_labels, v_indices = [], [], []

            if mode == 'train':
                num_samples = list(self.n_samples)
                num_val = [np.round(0.1 * num_samples[0] + 1e-6)] * 6

            else:
                num_val = [0] * self.n_classes
                num_samples = [100000] * self.n_classes

            index = 0

            for line in lines:
                bef_labels = line.split(':')[0]
                bef_input = line[len(bef_labels) + 1:-1]

                fine_label = bef_input.split(' ')[0]
                aft_input = bef_input[len(fine_label) + 1:]
                aft_label = label_dict[bef_labels]

                if raw_text:
                    aft_input = aft_input
                else:
                    aft_input = self.tokenizer.encode(aft_input, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                             return_tensors='pt')[0]

                aft_label = torch.tensor(aft_label).long()

                if num_val_full[aft_label] > 0:
                    if num_val[aft_label] > 0:
                        v_inputs.append(aft_input)
                        v_labels.append(aft_label)
                        v_indices.append(index)

                        num_val[aft_label] -= 1
                    num_val_full[aft_label] -= 1
                elif num_samples[aft_label] > 0:
                    inputs.append(aft_input)
                    labels.append(aft_label)
                    indices.append(index)

                    num_samples[aft_label] -= 1
                index += 1

            if raw_text:
                dataset = zip(inputs, labels)
            else:
                dataset = create_tensor_dataset(inputs, labels, indices)

            if mode == 'train':
                v_dataset = create_tensor_dataset(v_inputs, v_labels, v_indices)
                return dataset, v_dataset
            else:
                return dataset


class GLUEDataset(BaseDataset):
    def __init__(self, data_name, n_class, tokenizer, data_ratio=1.0, seed=0):
        super(GLUEDataset, self).__init__(data_name, n_class, tokenizer, data_ratio, seed)

        self.data_name = data_name
        self.data_ratio = data_ratio

    def _preprocess(self):
        print('Pre-processing news dataset...')
        train_dataset = self._load_dataset('train')

        if self.data_name == 'mnli':
            val_dataset = self._load_dataset('validation_matched')
            test_dataset = self._load_dataset('validation_mismatched')
        else:
            val_dataset = self._load_dataset('validation')
            test_dataset = val_dataset

        # Use the same dataset for validation and test
        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'validation', 'validation_matched', 'validation_mismatched']

        data_set = load_dataset('glue', self.data_name, split=mode)

        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        for i in range(min(int(self.data_ratio), len(data_set))):
            data_n = data_set[i]

            if self.data_name == 'cola' or self.data_name == 'sst2':
                toks = self.tokenizer.encode(data_n['sentence'], add_special_tokens=True, max_length=128,
                                             pad_to_max_length=True, return_tensors='pt')
            else:
                if self.data_name == 'qnli':
                    sent1, sent2 = data_n['question'], data_n['sentence']
                elif self.data_name == 'qqp':
                    sent1, sent2 = data_n['question1'], data_n['question2']
                elif self.data_name == 'mnli':
                    sent1, sent2 = data_n['premise'], data_n['hypothesis']
                else:  # wnli, rte, mrpc, stsb
                    sent1, sent2 = data_n['sentence1'], data_n['sentence2']
                toks = self.tokenizer.encode(sent1, sent2, add_special_tokens=True, max_length=128,
                                             pad_to_max_length=True, return_tensors='pt')

            if self.data_name == 'stsb':
                label = torch.tensor(data_n['label'])
            else:
                label = torch.tensor(data_n['label']).long()

            inputs.append(toks[0])
            labels.append(label)
            indices.append(i)

        dataset = create_tensor_dataset(inputs, labels, indices)
        return dataset