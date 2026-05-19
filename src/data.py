# -*- coding: utf-8 -*-
"""
极简 RNA Dataset —— 一次只处理一个样本，不做 upsampling、不做 batch 内合并

设计目标:
1. 把 binning cPickle 文件的所有样本扁平化为 (file_path, sample_idx) 列表
2. __getitem__ 只 unpickle 一次，只取一条样本
3. 把 padding、contact map、token 全部在 __getitem__ 里完成（CPU），FCN 特征留给 GPU 现场算
4. collate_fn 简单地 stack 多条样本（同 set_max_len），用以支持 batch_size > 1
5. 同长度桶内 batch 化，跨桶不 batch（避免 padding 浪费）

与 FastDataset 的区别:
- 不做 upsampling 时不会反复 unpickle
- 即使 upsampling，也只重复 (file_path, idx) 索引，不重复加载整个文件
- 单文件 cache：同一个 worker 拿到同一个 file 会复用上一次 unpickle 结果
"""

import os
import sys
import pickle as cPickle
from typing import List, Tuple
import numpy as np
import torch
from torch.utils import data

# 引入公共模块 (在 src/ 下)
SYMFOLD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (SYMFOLD_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.data_utils import seq_encoding
from datasets.data_generator import _CompatUnpickler, RNA_SS_data, generate_token_batch


def _round_up_80(x: int) -> int:
    return (x // 80 + int(x % 80 != 0)) * 80


def _list_pickles(root: str) -> List[str]:
    """递归列出 root 下所有 .cPickle 文件"""
    if os.path.isfile(root):
        return [root]
    if not os.path.isdir(root) and os.path.isfile(root + '.cPickle'):
        return [root + '.cPickle']
    files = []
    for r, _, fnames in sorted(os.walk(root)):
        for fname in sorted(fnames):
            if fname.endswith(('.cPickle', '.Pickle')):
                files.append(os.path.join(r, fname))
    return files


def build_index(roots: List[str], verbose: bool = True) -> List[Tuple[str, int, int]]:
    """
    扫描所有 cPickle 文件，构建 (file_path, sample_idx, seq_len) 索引

    返回: list of tuples，每条 = (绝对路径, 文件内下标, 序列长度)
    """
    files = []
    for root in roots:
        files += _list_pickles(root)
    if verbose:
        print(f'[Index] found {len(files)} pickle files under {len(roots)} roots')

    index = []
    for fi, fp in enumerate(files):
        try:
            with open(fp, 'rb') as f:
                items = _CompatUnpickler(f).load()
        except Exception as e:
            print(f'[Index] WARNING: failed to load {fp}: {e}')
            continue
        for si, item in enumerate(items):
            index.append((fp, si, int(item.length)))
        if verbose and (fi + 1) % 200 == 0:
            print(f'[Index] scanned {fi + 1}/{len(files)} files, {len(index)} samples so far')
    if verbose:
        print(f'[Index] total samples = {len(index)}')
    return index


def pairs_to_contact(pairs, seq_len: int, target_len: int) -> np.ndarray:
    out = np.zeros((target_len, target_len), dtype=np.float32)
    if pairs is None:
        return out
    for p in pairs:
        i, j = int(p[0]), int(p[1])
        if 0 <= i < seq_len and 0 <= j < seq_len:
            out[i, j] = 1.0
    return out


def encode_one_sample(item, set_max_len: int):
    """
    将一条 namedtuple 样本编码为模型输入张量

    支持两种格式:
      A) namedtuple('RNA_SS_data', 'seq seq_raw length name pairs')        - raw
      B) namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name') - preprocessed

    返回:
      contact:   (1, L, L) float32   — L = set_max_len
      seq_oh:    (L, 4)    float32
      seq_raw:   str
      length:    int
      name:      str
    """
    L = set_max_len
    seq_len = int(item.length)
    is_raw = hasattr(item, 'pairs')

    # --- one-hot 序列 ---
    seq_oh = np.zeros((L, 4), dtype=np.float32)
    if hasattr(item, 'seq') and getattr(item, 'seq', None) is not None:
        raw_oh = np.array(item.seq, dtype=np.float32)
        cp = min(seq_len, raw_oh.shape[0])
        seq_oh[:cp, :] = raw_oh[:cp, :]
    else:
        # 从 seq_raw 重新编码
        enc = seq_encoding(item.seq_raw).astype(np.float32)
        cp = min(seq_len, enc.shape[0])
        seq_oh[:cp, :] = enc[:cp, :]

    # --- contact map ---
    contact = np.zeros((L, L), dtype=np.float32)
    if is_raw:
        sub = pairs_to_contact(item.pairs, seq_len, seq_len)
        contact[:seq_len, :seq_len] = sub
    else:
        c = np.array(item.contact, dtype=np.float32)
        ch, cw = c.shape
        ch = min(ch, L)
        cw = min(cw, L)
        contact[:ch, :cw] = c[:ch, :cw]

    return contact, seq_oh, str(item.seq_raw), seq_len, str(item.name)


class SimpleRNADataset(data.Dataset):
    """
    简化版 Dataset: 每个 index 对应一条样本（不是一个文件）

    支持:
      - cache_file: 单 worker 内缓存上一次打开的 cPickle，避免反复 unpickle
      - 同长度桶 batch_sampler 由外部 BucketBatchSampler 处理（保持 set_max_len 一致）
    """

    def __init__(self, index: List[Tuple[str, int, int]]):
        self.index = index
        self._cache_path = None
        self._cache_items = None

    def __len__(self):
        return len(self.index)

    def _load_file(self, path: str):
        if self._cache_path != path:
            with open(path, 'rb') as f:
                self._cache_items = _CompatUnpickler(f).load()
            self._cache_path = path
        return self._cache_items

    def __getitem__(self, idx):
        path, si, seq_len = self.index[idx]
        items = self._load_file(path)
        item = items[si]
        # set_max_len 由 BucketBatchSampler 保证桶内一致
        # 这里我们只算到该样本的 80 倍数即可
        target_len = _round_up_80(seq_len)
        contact, seq_oh, seq_raw, length, name = encode_one_sample(item, target_len)
        return {
            'contact': contact,         # np (L, L)
            'seq_oh': seq_oh,           # np (L, 4)
            'seq_raw': seq_raw,         # str
            'length': length,           # int
            'name': name,               # str
            'set_max_len': target_len,  # int
        }


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    按 set_max_len（80 倍数）分桶，桶内随机洗牌后按 batch_size 切

    优势:
      - 同 batch 内 set_max_len 一致，避免再做跨长度 padding
      - 每个 epoch 都重新洗牌
    """

    def __init__(self, index: List[Tuple[str, int, int]],
                 batch_size_table: dict = None,
                 shuffle: bool = True,
                 max_set_len: int = 9999,
                 seed: int = 0):
        """
        batch_size_table: dict, key = set_max_len(80倍数), value = batch_size
                          未列出的桶默认 batch_size=1
        max_set_len: 超过这个长度的样本直接丢弃，防 OOM
        """
        self.index = index
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.max_set_len = max_set_len
        # 默认按桶大小给不同 batch
        self.bs_table = batch_size_table or {
            80: 16, 160: 8, 240: 4, 320: 2, 400: 2, 480: 2, 560: 1, 640: 1,
        }

        # 按 set_max_len 分桶
        self.buckets = {}
        for i, (_, _, seq_len) in enumerate(self.index):
            sm = _round_up_80(seq_len)
            if sm > self.max_set_len:
                continue
            self.buckets.setdefault(sm, []).append(i)

        # 计算总 batch 数
        self._batches = self._build_batches(seed=self.seed)

    def _bs_for(self, set_max_len: int) -> int:
        return self.bs_table.get(set_max_len, 1)

    def _build_batches(self, seed: int):
        rng = np.random.default_rng(seed)
        batches = []
        for sm, idxs in self.buckets.items():
            order = list(idxs)
            if self.shuffle:
                rng.shuffle(order)
            bs = self._bs_for(sm)
            for k in range(0, len(order), bs):
                batches.append(order[k:k + bs])
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._batches = self._build_batches(seed=self.seed + epoch)

    def __iter__(self):
        for b in self._batches:
            yield b

    def __len__(self):
        return len(self._batches)

    def stats(self) -> str:
        parts = [f'{sm}:{len(v)}samples/{((len(v) - 1)//self._bs_for(sm)) + 1}batches'
                 for sm, v in sorted(self.buckets.items())]
        return ' | '.join(parts)


def simple_collate_fn(batch_items, alphabet):
    """
    把同长度桶内的样本 stack 起来
    """
    set_max_len = max(b['set_max_len'] for b in batch_items)

    contact = np.stack([b['contact'] for b in batch_items], axis=0)  # (B, L, L)
    seq_oh = np.stack([b['seq_oh'] for b in batch_items], axis=0)    # (B, L, 4)
    lengths = np.array([b['length'] for b in batch_items], dtype=np.int64)
    seq_raws = [b['seq_raw'] for b in batch_items]
    names = [b['name'] for b in batch_items]

    contact_t = torch.from_numpy(contact).unsqueeze(1).float()  # (B, 1, L, L)
    seq_oh_t = torch.from_numpy(seq_oh).float()                 # (B, L, 4)
    length_t = torch.from_numpy(lengths).long()
    seq_enc_t = seq_oh_t.clone()                                # 与 seq_oh 等价

    tokens = generate_token_batch(alphabet, seq_raws)           # (B, L+2)

    return {
        'contact': contact_t,
        'seq_oh': seq_oh_t,
        'seq_enc': seq_enc_t,
        'tokens': tokens,
        'length': length_t,
        'seq_raws': seq_raws,
        'names': names,
        'set_max_len': set_max_len,
    }
