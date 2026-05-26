# -*- coding: utf-8 -*-
"""
SymFold v5 训练入口

基于 v4 trainer，改动:
- 使用 SymFoldModel_v5 (Wider FM + Density Conditioning + Output Refine)
- 默认配置: train_config_v5.json

用法:
    cd /root/aigame/dannyyan/RNADiffFold/symfold
    python -u train/train_v5.py train/config/train_config_v5.json
"""
from __future__ import annotations

import sys
import os
import json
import time
import signal
import logging
import traceback
import faulthandler
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

# 路径
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
SYMFOLD_ROOT = os.path.dirname(TRAIN_DIR)
SYMFOLD_SRC = os.path.join(SYMFOLD_ROOT, 'src')
for p in (SYMFOLD_ROOT, SYMFOLD_SRC,
          os.path.join(SYMFOLD_SRC, 'models', 'condition', 'fm_conditioner')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.v5.model import SymFoldModel_v5
from src.data import (SimpleRNADataset, BucketBatchSampler,
                       build_index, simple_collate_fn)
from src.gpu_features import get_data_fcn_gpu
from common.data_utils import contact_map_masks
from common.loss_utils import rna_evaluation


# 完整评估集：训练中每隔 N 个 epoch 跑一次，用于追踪泛化表现
FULL_EVAL_SET_FILES = {
    'bpRNA':        ['data/bpRNA/TS0.cPickle'],
    'RNAStrAlign':  ['data/RNAStrAlign/test.cPickle'],
    'bpRNA-new':    ['data/bpRNA-new/bpRNAnew.cPickle'],
    'ArchiveII':    ['data/ArchiveII/archiveII.cPickle'],
    'PDB_TS1':      ['data/PDB/TS1.cPickle'],
    'PDB_TS2':      ['data/PDB/TS2.cPickle'],
    'PDB_TS3':      ['data/PDB/TS3.cPickle'],
    'PDB_TS_hard':  ['data/PDB/TS_hard.cPickle'],
}


# ============================================================
# 信号处理 / 日志
# ============================================================

def install_signal_handlers(logger, heartbeat_path):
    def _dump(sig, frame):
        logger.error(f'>>> received signal {sig}, dumping stack <<<')
        try:
            faulthandler.dump_traceback(file=sys.stderr)
        except Exception:
            pass
        try:
            with open(heartbeat_path + '.signal', 'w') as f:
                f.write(f'signal={sig} time={time.asctime()}\n')
        except Exception:
            pass
        if sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            sys.exit(128 + sig)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP,
                 signal.SIGUSR1, signal.SIGUSR2):
        try:
            signal.signal(sig, _dump)
        except Exception:
            pass
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass


def setup_logging(config):
    task = config['task_name']
    log_dir = os.path.join(SYMFOLD_ROOT, config['paths']['log_dir'], task)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{task}.log')
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout),
        ])
    return logging.getLogger('SymFold-v5')


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_heartbeat(path, payload):
    try:
        with open(path, 'w') as f:
            f.write(json.dumps(payload, default=str))
    except Exception:
        pass


# ============================================================
# 数据
# ============================================================

def build_dataloaders(config, alphabet, logger):
    tcfg = config['training']
    pcfg = config['paths']

    name = tcfg['dataset']
    if name == 'standard':
        # 标准训练集: bpRNA TR0 + RNAStrAlign train
        # 验证集: bpRNA VL0 + RNAStrAlign val
        train_roots = [
            os.path.join(pcfg['preprocess_root'], 'RNAStrAlign'),
            os.path.join(pcfg['preprocess_root'], 'bpRNA'),
        ]
        val_roots = [
            os.path.join(pcfg['data_root'], 'bpRNA', 'VL0.cPickle'),
            os.path.join(pcfg['data_root'], 'RNAStrAlign', 'val.cPickle'),
        ]
    elif name == 'all':
        # 兼容旧配置: bpRNA TR0 + RNAStrAlign + bpRNA-new
        train_roots = [
            os.path.join(pcfg['preprocess_root'], 'RNAStrAlign'),
            os.path.join(pcfg['preprocess_root'], 'bpRNA'),
            os.path.join(pcfg['preprocess_root'], 'bpRNA-new'),
        ]
        val_roots = [os.path.join(pcfg['data_root'], 'bpRNA', 'VL0.cPickle')]
    elif name == 'bpRNA_only':
        train_roots = [os.path.join(pcfg['preprocess_root'], 'bpRNA')]
        val_roots = [os.path.join(pcfg['data_root'], 'bpRNA', 'VL0.cPickle')]
    else:
        raise ValueError(f'Unknown dataset: {name}')

    train_idx = build_index(train_roots, verbose=False)
    val_idx = build_index(val_roots, verbose=False)
    logger.info(f'[Data] train_samples={len(train_idx)}  val_samples={len(val_idx)}')

    bs_table = tcfg.get('bucket_batch_size')
    if bs_table is not None:
        bs_table = {int(k): int(v) for k, v in bs_table.items()}

    max_set_len = tcfg.get('max_set_len', 640)
    train_sampler = BucketBatchSampler(
        train_idx, batch_size_table=bs_table, shuffle=True,
        max_set_len=max_set_len, seed=config['seed'])
    val_sampler = BucketBatchSampler(
        val_idx, batch_size_table=bs_table, shuffle=False,
        max_set_len=max_set_len, seed=config['seed'])
    logger.info(f'[Data] buckets: {train_sampler.stats()}')
    logger.info(f'[Data] batches/epoch={len(train_sampler)} val_batches={len(val_sampler)}')

    collate = partial(simple_collate_fn, alphabet=alphabet)
    train_loader = DataLoader(
        SimpleRNADataset(train_idx), batch_sampler=train_sampler, collate_fn=collate,
        num_workers=tcfg.get('num_workers', 0),
        pin_memory=tcfg.get('pin_memory', False),
        persistent_workers=False)
    val_loader = DataLoader(
        SimpleRNADataset(val_idx), batch_sampler=val_sampler, collate_fn=collate,
        num_workers=min(tcfg.get('num_workers', 0), 2),
        pin_memory=tcfg.get('pin_memory', False),
        persistent_workers=False)
    return train_loader, val_loader


# ============================================================
# 可视化
# ============================================================

def visualize(gt, pred, seq_len, name, epoch, metrics, save_path):
    gt = gt[:seq_len, :seq_len]
    pred = pred[:seq_len, :seq_len]
    acc, prec, rec, sens, spec, f1, mcc = metrics
    fig = plt.figure(figsize=(21, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(gt, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title(f'GT ({int((gt > 0.5).sum())//2} pairs)')
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(pred, cmap='Oranges', vmin=0, vmax=1)
    ax2.set_title(f'Pred ({int((pred > 0.5).sum())//2} pairs)')
    ax3 = fig.add_subplot(gs[2])
    overlay = np.ones((seq_len, seq_len, 3)) * 0.95
    gt_b = gt > 0.5; pr_b = pred > 0.5
    tp = gt_b & pr_b; fn = gt_b & ~pr_b; fp = ~gt_b & pr_b
    overlay[tp] = [0.2, 0.8, 0.2]
    overlay[fn] = [0.9, 0.2, 0.2]
    overlay[fp] = [0.2, 0.4, 0.9]
    ax3.imshow(overlay)
    ax3.set_title(f'TP={int(tp.sum())//2} FN={int(fn.sum())//2} FP={int(fp.sum())//2}')
    ax3.legend(handles=[
        Patch(facecolor=(0.2, 0.8, 0.2), label='TP'),
        Patch(facecolor=(0.9, 0.2, 0.2), label='FN'),
        Patch(facecolor=(0.2, 0.4, 0.9), label='FP'),
    ], loc='lower right', fontsize=8)
    fig.suptitle(f'e{epoch} {name} L={seq_len} F1={f1:.4f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================
# 训练曲线绘制
# ============================================================

def plot_curves(history, output_dir):
    try:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        epochs = [h['epoch'] + 1 for h in history]
        losses = [h['loss'] for h in history]
        axes[0].plot(epochs, losses, 'b-')
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')

        eval_e = [h['epoch'] + 1 for h in history if 'val_f1' in h]
        eval_f1 = [h['val_f1'] for h in history if 'val_f1' in h]
        if eval_f1:
            axes[1].plot(eval_e, eval_f1, 'g-o')
            axes[1].set_title(f'Val F1 (best={max(eval_f1):.4f})')
        axes[1].set_xlabel('Epoch')

        full_e = [h['epoch'] + 1 for h in history if 'full_eval_avg_f1' in h]
        full_f1 = [h['full_eval_avg_f1'] for h in history if 'full_eval_avg_f1' in h]
        if full_f1:
            axes[2].plot(full_e, full_f1, 'm-o')
            axes[2].set_title(f'Full Eval Avg F1 (best={max(full_f1):.4f})')
        axes[2].set_xlabel('Epoch')

        times = [h.get('time_s', 0) for h in history]
        if any(t > 0 for t in times):
            axes[3].plot(epochs, times, 'r-')
            axes[3].set_title('Epoch Time (s)')
        axes[3].set_xlabel('Epoch')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'curves.png'), dpi=100)
        plt.close()
    except Exception:
        pass


# ============================================================
# 训练 / 验证
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch, config, logger,
                    heartbeat_path, model_dir, global_state):
    model.train()
    tcfg = config['training']
    log_every = tcfg.get('log_every', 20)
    heartbeat_every = tcfg.get('heartbeat_every', 10)
    save_every_step = tcfg.get('save_every_step', 0)
    grad_clip = tcfg.get('grad_clip', 1.0)

    total_loss = 0.0
    total_bce = 0.0
    total_stack = 0.0
    total_nc = 0.0
    total_density = 0.0
    n = 0
    epoch_total = len(loader)
    t_start = time.time()
    logger.info(f'[Train] === e{epoch} starts ({epoch_total} batches) ===')

    for bi, batch in enumerate(loader):
        global_state['batch_idx'] = bi
        global_state['epoch'] = epoch
        t0 = time.time()
        try:
            contact = batch['contact'].to(device, non_blocking=True)
            seq_oh = batch['seq_oh'].to(device, non_blocking=True)
            seq_enc = batch['seq_enc'].to(device, non_blocking=True)
            tokens = batch['tokens'].to(device, non_blocking=True)
            length = batch['length'].to(device, non_blocking=True)
            set_max_len = int(batch['set_max_len'])

            with torch.no_grad():
                data_fcn_2 = get_data_fcn_gpu(seq_oh, set_max_len)

            matrix_rep = torch.zeros_like(contact)
            contact_masks = contact_map_masks(length, matrix_rep).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, ld = model(contact, data_fcn_2, tokens, contact_masks,
                              set_max_len, seq_enc)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f'[Train] e{epoch} b{bi} NaN/Inf, skip')
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            l = float(loss.item())
            total_loss += l
            total_bce += float(ld.get('bce', 0))
            total_stack += float(ld.get('stack', 0))
            total_nc += float(ld.get('nc', 0))
            total_density += float(ld.get('density', 0))
            n += 1
            global_state['last_loss'] = l
            global_state['avg_loss'] = total_loss / n

            if bi % log_every == 0:
                logger.info(
                    f'[Train] e{epoch} b{bi}/{epoch_total} L={set_max_len} '
                    f'bs={contact.shape[0]} loss={l:.6f} '
                    f'bce={ld.get("bce", 0):.5f} stack={ld.get("stack", 0):.5f} '
                    f'nc={ld.get("nc", 0):.5f} den={ld.get("density", 0):.5f} '
                    f't={time.time()-t0:.3f}s')

            if bi % heartbeat_every == 0:
                write_heartbeat(heartbeat_path, {
                    'time': time.asctime(),
                    'unix_ts': time.time(),
                    'epoch': epoch, 'batch': bi,
                    'set_max_len': set_max_len,
                    'last_loss': l,
                    'avg_loss': total_loss / n,
                    'gpu_mb': torch.cuda.memory_allocated(device) / 1024 / 1024,
                    'pid': os.getpid(),
                })

            if save_every_step and (bi + 1) % save_every_step == 0:
                torch.save({
                    'epoch': epoch, 'batch': bi,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }, os.path.join(model_dir, 'last.pt'))

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f'[OOM] e{epoch} b{bi} L={set_max_len} skip')
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            logger.error(f'[Train] RuntimeError: {e}')
            logger.error(traceback.format_exc())
            raise

    avg = total_loss / max(n, 1)
    logger.info(f'[Train] === e{epoch} done: avg_loss={avg:.6f} '
                f'bce={total_bce/max(n,1):.5f} stack={total_stack/max(n,1):.5f} '
                f'nc={total_nc/max(n,1):.5f} den={total_density/max(n,1):.5f} '
                f'success={n}/{epoch_total} time={time.time()-t_start:.1f}s ===')
    return avg, time.time() - t_start


@torch.no_grad()
def evaluate(model, loader, device, config, logger, epoch, output_dir):
    model.eval()
    scfg = config['sampling']
    all_metrics = []
    vis_cnt = 0
    max_vis = config['training'].get('vis_samples', 5)
    logger.info(f'[Eval] === e{epoch} eval start ({len(loader)} batches) ===')
    t0 = time.time()

    for bi, batch in enumerate(loader):
        try:
            contact = batch['contact']
            seq_oh = batch['seq_oh'].to(device)
            seq_enc = batch['seq_enc'].to(device)
            tokens = batch['tokens'].to(device)
            length = batch['length'].to(device)
            set_max_len = int(batch['set_max_len'])
            data_fcn_2 = get_data_fcn_gpu(seq_oh, set_max_len)
            matrix_rep = torch.zeros_like(contact)
            contact_masks = contact_map_masks(length, matrix_rep).to(device)
            pred, prob = model.sample(
                data_fcn_2=data_fcn_2, tokens=tokens,
                contact_masks=contact_masks, set_max_len=set_max_len,
                seq_oh=seq_enc,
                num_steps=scfg.get('num_steps', 20),
                num_samples_per_input=scfg.get('num_samples_per_input', 1),
                physics_beta=scfg.get('physics_beta', 0.0),
                physics_lambda_pk=scfg.get('physics_lambda_pk', 0.0),
                physics_alpha_stack=scfg.get('physics_alpha_stack', 1.0),
            )
            pred = pred.cpu().float()
            for i in range(contact.shape[0]):
                m = rna_evaluation(pred[i].squeeze(), contact.float()[i].squeeze())
                all_metrics.append(m)
                if vis_cnt < max_vis:
                    name = batch['names'][i].replace('/', '_')
                    seq_len = int(length[i].item())
                    vp = os.path.join(output_dir, f'vis_e{epoch}_{name}.png')
                    visualize(contact.float()[i].squeeze().numpy(),
                              pred[i].squeeze().numpy(),
                              seq_len, name, epoch, m, vp)
                    vis_cnt += 1
            if bi % 20 == 0:
                logger.info(f'[Eval] e{epoch} b{bi}/{len(loader)} L={set_max_len}')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f'[Eval OOM] b{bi} skip')
                torch.cuda.empty_cache()
                continue
            raise

    if not all_metrics:
        return {}
    arr = np.array(all_metrics)
    acc, prec, rec, sens, spec, f1, mcc = [np.nan_to_num(arr[:, i]).mean() for i in range(7)]
    logger.info(f'[Eval] === e{epoch}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} '
                f'MCC={mcc:.4f} N={len(all_metrics)} time={time.time()-t0:.1f}s ===')
    return dict(f1=f1, precision=prec, recall=rec, mcc=mcc, N=len(all_metrics))


# ============================================================
# 完整 Eval + 报告
# ============================================================

def _resolve_eval_files(files, config):
    root = config['paths'].get('rnadifffold_root', os.path.dirname(SYMFOLD_ROOT))
    out = []
    for f in files:
        out.append(f if os.path.isabs(f) else os.path.join(root, f))
    return out


@torch.no_grad()
def evaluate_full_dataset(model, dataset_name, files, device, config, logger,
                          epoch, eval_dir, alphabet):
    """评估一个完整测试集，保存少量 GT/Pred 可视化。"""
    roots = _resolve_eval_files(files, config)
    idx = build_index(roots, verbose=False)
    if not idx:
        logger.warning(f'[FullEval] {dataset_name}: no samples found: {roots}')
        return None

    scfg = config['sampling']
    tcfg = config['training']
    bs_table = {int(k): int(v) for k, v in tcfg.get('full_eval_bucket_batch_size',
        {"80": 8, "160": 4, "240": 2, "320": 1, "400": 1, "480": 1, "560": 1, "640": 1}).items()}
    sampler = BucketBatchSampler(idx, batch_size_table=bs_table, shuffle=False,
                                 max_set_len=tcfg.get('max_set_len', 640), seed=0)
    collate = partial(simple_collate_fn, alphabet=alphabet)
    loader = DataLoader(SimpleRNADataset(idx), batch_sampler=sampler,
                        collate_fn=collate, num_workers=0,
                        pin_memory=tcfg.get('pin_memory', False))

    vis_dir = os.path.join(eval_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    max_vis = tcfg.get('full_eval_vis_samples_per_set', 3)
    vis_cnt = 0
    all_metrics = []
    samples = []
    t0 = time.time()
    logger.info(f'[FullEval] {dataset_name}: start N={len(idx)} batches={len(loader)}')

    for bi, batch in enumerate(loader):
        try:
            contact = batch['contact']
            seq_oh = batch['seq_oh'].to(device)
            seq_enc = batch['seq_enc'].to(device)
            tokens = batch['tokens'].to(device)
            length = batch['length'].to(device)
            set_max_len = int(batch['set_max_len'])
            data_fcn_2 = get_data_fcn_gpu(seq_oh, set_max_len)
            matrix_rep = torch.zeros_like(contact)
            contact_masks = contact_map_masks(length, matrix_rep).to(device)
            pred, _ = model.sample(
                data_fcn_2=data_fcn_2, tokens=tokens,
                contact_masks=contact_masks, set_max_len=set_max_len,
                seq_oh=seq_enc,
                num_steps=scfg.get('num_steps', 20),
                num_samples_per_input=scfg.get('num_samples_per_input', 1),
                physics_beta=scfg.get('physics_beta', 0.0),
                physics_lambda_pk=scfg.get('physics_lambda_pk', 0.0),
                physics_alpha_stack=scfg.get('physics_alpha_stack', 1.0),
                density_guided=scfg.get('density_guided', True),
            )
            pred = pred.cpu().float()
            for i in range(contact.shape[0]):
                m = rna_evaluation(pred[i].squeeze(), contact.float()[i].squeeze())
                all_metrics.append(m)
                acc, prec, rec, sens, spec, f1, mcc = [float(x) for x in m]
                seq_len = int(length[i].item())
                gt_pairs = int((contact.float()[i].squeeze()[:seq_len, :seq_len] > 0.5).sum().item()) // 2
                pred_pairs = int((pred[i].squeeze()[:seq_len, :seq_len] > 0.5).sum().item()) // 2
                samples.append({
                    'name': batch['names'][i], 'length': seq_len,
                    'f1': f1, 'precision': prec, 'recall': rec, 'mcc': mcc,
                    'gt_num_pairs': gt_pairs, 'pred_num_pairs': pred_pairs,
                })
                if vis_cnt < max_vis:
                    safe_name = batch['names'][i].replace('/', '_')
                    vp = os.path.join(vis_dir, f'e{epoch+1:03d}_{dataset_name}_{safe_name}.png')
                    visualize(contact.float()[i].squeeze().numpy(),
                              pred[i].squeeze().numpy(), seq_len,
                              f'{dataset_name}/{safe_name}', epoch + 1, m, vp)
                    vis_cnt += 1
            if bi % 20 == 0:
                logger.info(f'[FullEval] {dataset_name} b{bi}/{len(loader)} L={set_max_len}')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f'[FullEval OOM] {dataset_name} b{bi} skip')
                torch.cuda.empty_cache()
                continue
            raise

    if not all_metrics:
        return None
    arr = np.array(all_metrics)
    acc, prec, rec, sens, spec, f1, mcc = [np.nan_to_num(arr[:, i]).mean() for i in range(7)]
    samples.sort(key=lambda x: x['f1'])
    res = {
        'N': len(all_metrics),
        'F1': float(f1), 'Precision': float(prec), 'Recall': float(rec),
        'MCC': float(mcc), 'Accuracy': float(acc), 'Specificity': float(spec),
        'top_worst': samples[:10],
        'top_best': list(reversed(samples[-10:])),
    }
    logger.info(f'[FullEval] {dataset_name}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} '
                f'MCC={mcc:.4f} N={len(all_metrics)} time={time.time()-t0:.1f}s')
    return res


def plot_full_eval_bar(results, save_path, epoch):
    names = list(results.keys())
    f1s = [results[n]['F1'] for n in names]
    plt.figure(figsize=(12, 5))
    bars = plt.bar(names, f1s, color='#4C78A8')
    plt.ylim(0, 1)
    plt.ylabel('F1')
    plt.title(f'Full Eval F1 @ Epoch {epoch}')
    plt.xticks(rotation=30, ha='right')
    for b, v in zip(bars, f1s):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_full_eval_trend(summary_path, save_path):
    if not os.path.isfile(summary_path):
        return
    with open(summary_path, 'r') as f:
        hist = json.load(f)
    if not hist:
        return
    datasets = sorted({k for h in hist for k in h.get('results', {}).keys()})
    plt.figure(figsize=(12, 6))
    for ds in datasets:
        xs, ys = [], []
        for h in hist:
            if ds in h.get('results', {}):
                xs.append(h['epoch'])
                ys.append(h['results'][ds]['F1'])
        if ys:
            plt.plot(xs, ys, marker='o', label=ds)
    avg_x = [h['epoch'] for h in hist]
    avg_y = [h.get('avg_f1', 0) for h in hist]
    plt.plot(avg_x, avg_y, marker='s', linewidth=3, color='black', label='Average')
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('Full Eval F1 Trend')
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def write_full_eval_report(results, out_path, epoch, config, avg_f1):
    lines = []
    lines.append(f'# SymFold v5 Full Eval Report — Epoch {epoch}')
    lines.append('')
    lines.append(f'- 时间: {time.asctime()}')
    lines.append(f'- checkpoint: `{config["paths"]["model_save_dir"]}/last.pt`')
    lines.append(f'- sampling steps: `{config["sampling"].get("num_steps", 20)}`')
    lines.append(f'- density_guided: `{config["sampling"].get("density_guided", True)}`')
    lines.append(f'- Average F1: **{avg_f1:.4f}**')
    lines.append('')
    lines.append('## 数据集汇总')
    lines.append('')
    lines.append('| Dataset | N | F1 | Precision | Recall | MCC |')
    lines.append('|---------|---:|:--:|:---------:|:------:|:---:|')
    for name, r in sorted(results.items(), key=lambda kv: kv[1]['F1'], reverse=True):
        lines.append(f'| {name} | {r["N"]} | **{r["F1"]:.4f}** | {r["Precision"]:.4f} | {r["Recall"]:.4f} | {r["MCC"]:.4f} |')
    lines.append('')
    lines.append('## 最差样本 Top-5')
    for name, r in results.items():
        lines.append('')
        lines.append(f'### {name}')
        lines.append('| Sample | L | F1 | P | R | GT pairs | Pred pairs |')
        lines.append('|--------|--:|:--:|:--:|:--:|---------:|-----------:|')
        for s in r.get('top_worst', [])[:5]:
            lines.append(f'| `{s["name"]}` | {s["length"]} | {s["f1"]:.4f} | {s["precision"]:.4f} | {s["recall"]:.4f} | {s["gt_num_pairs"]} | {s["pred_num_pairs"]} |')
    lines.append('')
    lines.append('## 可视化文件')
    lines.append('')
    lines.append(f'- 本轮柱状图: `full_eval_f1_bar_e{epoch:03d}.png`')
    lines.append('- 趋势曲线: `../full_eval_f1_trend.png`')
    lines.append('- 样本可视化: `vis/e{epoch:03d}_*.png`')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


@torch.no_grad()
def run_full_eval(model, device, config, logger, epoch, output_dir, alphabet):
    tcfg = config['training']
    enabled = tcfg.get('full_eval_enabled', True)
    if not enabled:
        return None
    epoch_num = epoch + 1
    eval_root = os.path.join(output_dir, 'full_eval')
    eval_dir = os.path.join(eval_root, f'e{epoch_num:03d}')
    os.makedirs(eval_dir, exist_ok=True)

    selected = tcfg.get('full_eval_sets')
    if selected:
        names = [x.strip() for x in selected.split(',') if x.strip()]
    else:
        names = list(FULL_EVAL_SET_FILES.keys())

    was_training = model.training
    model.eval()
    results = {}
    logger.info(f'[FullEval] ===== epoch {epoch_num} full eval start: {names} =====')
    t0 = time.time()
    for name in names:
        if name not in FULL_EVAL_SET_FILES:
            logger.warning(f'[FullEval] skip unknown dataset: {name}')
            continue
        res = evaluate_full_dataset(model, name, FULL_EVAL_SET_FILES[name],
                                    device, config, logger, epoch, eval_dir, alphabet)
        if res is not None:
            results[name] = res

    if was_training:
        model.train()
    if not results:
        return None

    avg_f1 = float(np.mean([r['F1'] for r in results.values()]))
    out_json = os.path.join(eval_dir, f'full_eval_e{epoch_num:03d}.json')
    with open(out_json, 'w') as f:
        json.dump({'epoch': epoch_num, 'avg_f1': avg_f1, 'results': results}, f, indent=2)

    plot_full_eval_bar(results, os.path.join(eval_dir, f'full_eval_f1_bar_e{epoch_num:03d}.png'), epoch_num)
    report_path = os.path.join(eval_dir, f'FULL_EVAL_REPORT_e{epoch_num:03d}.md')
    write_full_eval_report(results, report_path, epoch_num, config, avg_f1)

    summary_path = os.path.join(eval_root, 'full_eval_history.json')
    hist = []
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, 'r') as f:
                hist = json.load(f)
        except Exception:
            hist = []
    hist = [h for h in hist if h.get('epoch') != epoch_num]
    hist.append({'epoch': epoch_num, 'avg_f1': avg_f1,
                 'results': {k: {'F1': v['F1'], 'N': v['N']} for k, v in results.items()}})
    hist.sort(key=lambda x: x['epoch'])
    with open(summary_path, 'w') as f:
        json.dump(hist, f, indent=2)
    plot_full_eval_trend(summary_path, os.path.join(eval_root, 'full_eval_f1_trend.png'))

    logger.info(f'[FullEval] ===== epoch {epoch_num} done avg_f1={avg_f1:.4f} '
                f'time={time.time()-t0:.1f}s report={report_path} =====')
    return {'avg_f1': avg_f1, 'results': results, 'report': report_path}


# ============================================================
# main
# ============================================================

def main():
    cfg_path = os.path.join(TRAIN_DIR, 'config', 'train_config_v5.json')
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    config = load_config(cfg_path)
    logger = setup_logging(config)

    try:
        os.setsid()
    except Exception:
        pass

    task = config['task_name']
    log_dir = os.path.join(SYMFOLD_ROOT, config['paths']['log_dir'], task)
    os.makedirs(log_dir, exist_ok=True)
    heartbeat_path = os.path.join(log_dir, f'{task}.heartbeat')
    install_signal_handlers(logger, heartbeat_path)
    write_heartbeat(heartbeat_path, {'event': 'start', 'time': time.asctime(),
                                      'pid': os.getpid()})

    logger.info('=' * 70)
    logger.info(f'SymFold v5 Training: {task}')
    logger.info(f'PID={os.getpid()} PGID={os.getpgrp()} py={sys.version.split()[0]}')
    logger.info(f'Torch={torch.__version__} CUDA={torch.version.cuda}')
    logger.info(f'config:\n{json.dumps(config, indent=2)}')
    logger.info('=' * 70)

    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    cpu_threads = config['training'].get('cpu_threads', 4)
    torch.set_num_threads(cpu_threads)
    os.environ.setdefault('OMP_NUM_THREADS', str(cpu_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(cpu_threads))

    model_dir = os.path.join(SYMFOLD_ROOT, config['paths']['model_save_dir'])
    output_dir = os.path.join(SYMFOLD_ROOT, config['paths']['output_dir'])
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger.info('[Init] building v5 model on CPU...')
    t0 = time.time()
    mc = config['model']
    model = SymFoldModel_v5(
        hidden_dim=mc['hidden_dim'], num_heads=mc['num_heads'],
        dim_head=mc['dim_head'], num_layers=mc['num_layers'],
        patch_size=mc['patch_size'], cond_dim=mc['cond_dim'],
        max_len=mc['max_len'], dp_rate=mc['dp_rate'],
        rho_0=mc['rho_0'],
        pos_weight_base=mc.get('pos_weight_base', 199.0),
        pos_weight_min=mc.get('pos_weight_min', 20.0),
        focal_gamma=mc.get('focal_gamma', 1.5),
        u_ckpt=mc['u_conditioner_ckpt'],
        num_families=mc.get('num_families', 0),
        dilation_pattern=mc.get('dilation_pattern'),
        stack_weight=mc.get('stack_weight', 0.05),
        nc_weight=mc.get('nc_weight', 0.02),
        density_weight=mc.get('density_weight', 0.2),
        tri_start_layer=mc.get('tri_start_layer', 6),
        tri_dim=mc.get('tri_dim', 64),
        fm_multi_out_dim=mc.get('fm_multi_out_dim', 64),
    )
    logger.info(f'[Init] built in {time.time()-t0:.1f}s')
    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[Init] params total={total:,} trainable={train_p:,}')

    alphabet = model.get_alphabet()
    train_loader, val_loader = build_dataloaders(config, alphabet, logger)

    logger.info(f'[Init] move to {device}')
    model.to(device)
    logger.info(f'[Init] gpu_alloc={torch.cuda.memory_allocated(device)/1024/1024:.0f}MB')

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training']['lr'], weight_decay=0.01)

    # ---- resume ----
    start_epoch = 0
    best_f1 = 0.0
    history = []
    last_ckpt = os.path.join(model_dir, 'last.pt')
    if config['training'].get('auto_resume', True) and os.path.isfile(last_ckpt):
        try:
            ck = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            start_epoch = ck.get('epoch', 0) + 1
            best_f1 = ck.get('best_f1', 0.0) or 0.0
            history = ck.get('history', [])

            history_json = os.path.join(output_dir, 'history.json')
            if not history and os.path.isfile(history_json):
                try:
                    with open(history_json, 'r') as hf:
                        history = json.load(hf)
                    logger.info(f'[Resume] recovered history ({len(history)} entries)')
                except Exception as he:
                    logger.warning(f'[Resume] failed to load history.json: {he}')

            hist_f1s = [h.get('val_f1', 0) for h in history if 'val_f1' in h]
            if hist_f1s:
                best_f1 = max(best_f1, max(hist_f1s))

            logger.info(f'[Resume] e={start_epoch} best_f1={best_f1:.4f} history={len(history)} entries')
        except Exception as e:
            logger.warning(f'[Resume] failed: {e}, from scratch')

    tcfg = config['training']
    eval_every = tcfg.get('eval_every', 2)
    full_eval_every = tcfg.get('full_eval_every', 20)
    save_every = tcfg.get('save_every', 5)
    patience = tcfg.get('patience', 30)
    patience_cnt = 0
    global_state = {'epoch': start_epoch, 'batch_idx': -1,
                     'last_loss': 0, 'avg_loss': 0}

    def warmup_cosine_lr(e):
        warmup = tcfg['warmup_epochs']
        total_e = tcfg['epochs']
        base_lr = tcfg['lr']
        if e < warmup:
            return base_lr * (e + 1) / warmup
        import math
        progress = (e - warmup) / max(total_e - warmup, 1)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    try:
        for epoch in range(start_epoch, tcfg['epochs']):
            lr = warmup_cosine_lr(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            logger.info(f'[LR] e{epoch}: {lr:.6g}')

            avg, epoch_time = train_one_epoch(
                model, train_loader, optimizer, device, epoch,
                config, logger, heartbeat_path, model_dir, global_state)

            h_entry = {'epoch': epoch, 'loss': avg, 'time_s': epoch_time, 'lr': lr}

            torch.save({
                'epoch': epoch, 'batch': -1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'avg_loss': avg, 'best_f1': best_f1,
                'history': history,
                'config': config,
            }, last_ckpt)

            if full_eval_every and (epoch + 1) % full_eval_every == 0:
                full_res = run_full_eval(model, device, config, logger,
                                         epoch, output_dir, alphabet)
                if full_res:
                    h_entry['full_eval_avg_f1'] = full_res['avg_f1']
                    for ds_name, ds_res in full_res['results'].items():
                        h_entry[f'full_eval_{ds_name}_f1'] = ds_res['F1']
                    h_entry['full_eval_report'] = full_res['report']

            if (epoch + 1) % eval_every == 0:
                res = evaluate(model, val_loader, device, config, logger,
                                epoch, output_dir)
                f1 = res.get('f1', 0)
                h_entry['val_f1'] = f1
                h_entry['val_prec'] = res.get('precision', 0)
                h_entry['val_rec'] = res.get('recall', 0)
                h_entry['val_mcc'] = res.get('mcc', 0)

                if f1 > best_f1:
                    best_f1 = f1
                    patience_cnt = 0
                    torch.save({'epoch': epoch, 'model': model.state_dict(),
                                 'val_f1': best_f1, 'config': config},
                                os.path.join(model_dir, 'best.pt'))
                    logger.info(f'[Save] new best F1={best_f1:.4f}')
                else:
                    patience_cnt += 1
                    logger.info(f'[Eval] no improve {patience_cnt}/{patience}')
                    if patience_cnt >= patience:
                        logger.info('[Eval] early stop')
                        history.append(h_entry)
                        break

            history.append(h_entry)

            with open(os.path.join(output_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            plot_curves(history, output_dir)

            if (epoch + 1) % save_every == 0:
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                             'config': config},
                            os.path.join(model_dir, f'epoch_{epoch}.pt'))

    except KeyboardInterrupt:
        logger.info('interrupted')
    except Exception as e:
        logger.error(f'[Fatal] {e}')
        logger.error(traceback.format_exc())

    write_heartbeat(heartbeat_path, {'event': 'end', 'time': time.asctime(),
                                      'pid': os.getpid(), 'best_f1': best_f1})
    logger.info(f'[Done] best_f1={best_f1:.4f}  {time.asctime()}')


if __name__ == '__main__':
    main()
