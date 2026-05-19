# -*- coding: utf-8 -*-
import collections
import os
import sys
import pickle as cPickle
from os.path import join
from random import shuffle
from torch.utils import data
from itertools import product
from typing import List, Tuple
from common.data_utils import *


perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1, 3], [3, 1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]

# namedtuple for raw data format
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq seq_raw length name pairs')


class _CompatUnpickler(cPickle.Unpickler):
    """Custom unpickler that redirects __main__.RNA_SS_data to our module-level definition."""
    def find_class(self, module, name):
        if name == 'RNA_SS_data':
            return RNA_SS_data
        return super().find_class(module, name)


def make_dataset(
        directory: str
) -> List[str]:
    instances = []
    directory = os.path.expanduser(directory)
    # Support direct file path (with or without extension)
    if os.path.isfile(directory):
        instances.append(directory)
        return instances
    if not os.path.isdir(directory) and os.path.isfile(directory + '.cPickle'):
        instances.append(directory + '.cPickle')
        return instances
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):
            if fname.endswith('.cPickle') or fname.endswith('.Pickle'):
                path = os.path.join(root, fname)
                instances.append(path)

    return instances


class ParserData(object):
    def __init__(self, path):
        self.path = path
        self.raw_data = self.load_data(self.path)
        self.len = len(self.raw_data)
        self.seq_max_len = max([x.length for x in self.raw_data])
        self.set_max_len = (self.seq_max_len // 80 + int(self.seq_max_len % 80 != 0)) * 80

    def load_data(self, path):
        with open(path, 'rb') as f:
            load_data = _CompatUnpickler(f).load()
        return load_data

    def padding(self, data_array, maxlen):
        a, b = data_array.shape
        return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')

    def pairs2map(self, pairs, seq_len):
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            if pair[0] < seq_len and pair[1] < seq_len:
                contact[pair[0], pair[1]] = 1
        return contact

    def _get_data_fcn(self, data_seq, data_length, set_length):
        """Generate 17-channel FCN features from one-hot encoded sequence."""
        data_fcn = np.zeros((16, set_length, set_length))
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_length, :data_length] = np.matmul(
                data_seq[:data_length, i].reshape(-1, 1),
                data_seq[:data_length, j].reshape(1, -1)
            )
        data_fcn_1 = np.zeros((1, set_length, set_length))
        data_fcn_1[0, :data_length, :data_length] = creatmat(data_seq[:data_length, :])
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return data_fcn_2

    def _is_raw_format(self):
        """Check if data is in raw format by checking if it has 'pairs' field."""
        item = self.raw_data[0]
        return hasattr(item, 'pairs')

    def preprocess_data(self):
        shuffle(self.raw_data)

        if self._is_raw_format():
            # Raw format: fields are (seq, seq_raw, length, name, pairs)
            contact_list = []
            data_fcn_2_list = []
            data_seq_raw_list = []
            data_length_list = []
            data_name_list = []

            for item in self.raw_data:
                seq_len = item.length
                one_hot_seq = np.array(item.seq)

                # Pad one-hot sequence to set_max_len
                padded_seq = np.zeros((self.set_max_len, 4))
                padded_seq[:seq_len, :] = one_hot_seq[:seq_len, :]

                # Generate contact map and pad
                contact = self.pairs2map(item.pairs, seq_len)
                contact_padded = np.zeros((self.set_max_len, self.set_max_len))
                contact_padded[:seq_len, :seq_len] = contact

                # Generate 17-channel FCN features
                data_fcn_2 = self._get_data_fcn(padded_seq, seq_len, self.set_max_len)

                contact_list.append(contact_padded)
                data_fcn_2_list.append(data_fcn_2)
                data_seq_raw_list.append(item.seq_raw)
                data_length_list.append(item.length)
                data_name_list.append(item.name)
        else:
            # Processed format: fields are (contact, data_fcn_2, seq_raw, length, name)
            contact_list = [item.contact for item in self.raw_data]
            data_fcn_2_list = [item.data_fcn_2 for item in self.raw_data]
            data_seq_raw_list = [item.seq_raw for item in self.raw_data]
            data_length_list = [item.length for item in self.raw_data]
            data_name_list = [item.name for item in self.raw_data]

        contact_array = np.stack(contact_list, axis=0)
        data_fcn_2_array = np.stack(data_fcn_2_list, axis=0)

        data_seq_encode_list = list(map(lambda x: seq_encoding(x), data_seq_raw_list))
        data_seq_encode_pad_list = list(map(lambda x: self.padding(x, self.set_max_len), data_seq_encode_list))
        data_seq_encode_pad_array = np.stack(data_seq_encode_pad_list, axis=0)

        return contact_array, data_fcn_2_array, data_seq_raw_list, data_length_list, data_name_list, self.set_max_len, data_seq_encode_pad_array


class Dataset(data.Dataset):

    def __init__(
            self,
            data_root: List[str],
            upsampling: bool = False
    ) -> None:
        self.data_root = data_root
        self.upsampling = upsampling
        if len(self.data_root) == 1:
            samples = self.make_dataset(self.data_root[0])
        elif len(self.data_root) > 1:
            samples = []
            for root in self.data_root:
                samples += self.make_dataset(root)
        else:
            raise ValueError('data_root is empty')

        self.samples = samples
        if self.upsampling:
            self.samples = self.upsampling_data()

    @staticmethod
    def make_dataset(
            directory: str
    ) -> List[str]:
        return make_dataset(directory)

    # for data balance, 4 times for 160~320 & 320~640
    def upsampling_data(self):
        augment_data_list = list()
        final_data_list = self.samples
        for data_path in final_data_list:
            with open(data_path, 'rb') as f:
                load_data = _CompatUnpickler(f).load()
            max_len = max([x.length for x in load_data])
            if max_len == 160:
                continue
            elif max_len == 320:
                augment_data_list.append(data_path)
            elif max_len == 640:
                augment_data_list.append(data_path)

        augment_data_list = list(np.random.choice(augment_data_list, 3 * len(augment_data_list)))
        final_data_list.extend(augment_data_list)
        shuffle(final_data_list)
        return final_data_list

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index: int):
        batch_data_path = self.samples[index]
        batch_data = ParserData(batch_data_path)

        contact_array, data_fcn_2_array, data_seq_raw_list, data_length_list, data_name_list, set_max_len, \
        data_seq_encode_pad_array = batch_data.preprocess_data()

        contact = torch.tensor(contact_array).unsqueeze(1).long()
        data_fcn_2 = torch.tensor(data_fcn_2_array).float()
        data_length = torch.tensor(data_length_list).long()
        data_seq_encode_pad = torch.tensor(data_seq_encode_pad_array).float()

        return contact, data_fcn_2, data_seq_raw_list, data_length, data_name_list, set_max_len, data_seq_encode_pad


def generate_token_batch(alphabet, seq_strs):
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len
            + int(alphabet.prepend_bos)
            + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    for i, seq_str in enumerate(seq_strs):
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[i, int(alphabet.prepend_bos): len(seq_str) + int(alphabet.prepend_bos), ] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_str) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens


def diff_collate_fn(batch, alphabet):
    contact, data_fcn_2, data_seq_raw_list, data_length, data_name_list, set_max_len, data_seq_encode_pad = zip(*batch)
    if len(contact) == 1:
        contact = contact[0]
        data_fcn_2 = data_fcn_2[0]
        data_seq_raw = data_seq_raw_list[0]
        data_length = data_length[0]
        data_name = data_name_list[0]
        set_max_len = set_max_len[0]
        data_seq_encode_pad = data_seq_encode_pad[0]

    else:
        set_max_len = max(set_max_len) if isinstance(set_max_len, tuple) else set_max_len

        contact_list = list()
        for item in contact:
            if item.shape[-1] < set_max_len:
                item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
                contact_list.append(item)
            else:
                contact_list.append(item)

        data_fcn_2_list = list()
        for item in data_fcn_2:
            if item.shape[-1] < set_max_len:
                item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
                data_fcn_2_list.append(item)
            else:
                data_fcn_2_list.append(item)

        data_seq_encode_pad_list = list()
        for item in data_seq_encode_pad:
            if item.shape[-1] < set_max_len:
                item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
                data_seq_encode_pad_list.append(item)
            else:
                data_seq_encode_pad_list.append(item)

        contact = torch.cat(contact_list, dim=0)
        data_fcn_2 = torch.cat(data_fcn_2_list, dim=0)
        data_seq_encode_pad = torch.cat(data_seq_encode_pad_list, dim=0)

        data_seq_raw = list()
        for item in data_seq_raw_list:
            data_seq_raw.extend(item)

        data_length = torch.cat(data_length, dim=0)

        data_name = list()
        for item in data_name_list:
            data_name.extend(item)

    tokens = generate_token_batch(alphabet, data_seq_raw)

    return contact, data_fcn_2, tokens, data_length, data_name, set_max_len, data_seq_encode_pad


def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')


def pairs2map(pairs, seq_len):
    contact = np.zeros([seq_len, seq_len])
    for pair in pairs:
        contact[pair[0], pair[1]] = 1
    return contact


def Gaussian(x):
    return math.exp(-0.5 * (x * x))


def paired(x, y):
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0


def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat
