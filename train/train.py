# -*- coding: utf-8 -*-
"""
SymFold 训练入口

设计 (吸取 v2 的踩坑教训):
- 全程 fp32 (避免 PyTorch 2.4.1 + H20 bf16 SIGFPE)
- faulthandler.enable() + 信号 handler 抓 C 层错误
- 心跳文件
- 独立 stderr
- 自动 checkpoint 恢复 (auto_resume)
- num_workers=0 默认 (排除 worker 隐患)
- setsid: 进程独立 PGID/SID, 不会被 codebuddy shell 杀

用法:
    cd /root/aigame/dannyyan/RNADiffFold/symfold
    setsid bash -c '
        source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260
        export PYTHONPATH=..
        exec python -u train/train.py
    ' < /dev/null \
        > logs/<task>.stdout.log \
        2> logs/<task>.stderr.log &
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

from src.v1.model import SymFoldModel
from src.data import (SimpleRNADataset, BucketBatchSampler,
                       build_index, simple_collate_fn)
from src.gpu_features import get_data_fcn_gpu
from common.data_utils import contact_map_masks
from common.loss_utils import rna_evaluation


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
    log_dir = os.path.join(SYMFOLD_ROOT, config['paths']['log_dir'])
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
    return logging.getLogger('SymFold')


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
    if name == 'all':
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
            n += 1
            global_state['last_loss'] = l
            global_state['avg_loss'] = total_loss / n

            if bi % log_every == 0:
                logger.info(
                    f'[Train] e{epoch} b{bi}/{epoch_total} L={set_max_len} '
                    f'bs={contact.shape[0]} loss={l:.6f} t={time.time()-t0:.3f}s')

            if bi % heartbeat_every == 0:
                write_heartbeat(heartbeat_path, {
                    'time': time.asctime(),
                    'unix_ts': time.time(),
                    'epoch': epoch, 'batch': bi,
                    'set_max_len': set_max_len,
                    'last_loss': l,
                    'avg_loss': total_loss / n,
                    'gpu_mb': torch.cuda.memory_allocated(device)/1024/1024,
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
                f'success={n}/{epoch_total} time={time.time()-t_start:.1f}s ===')
    return avg


@torch.no_grad()
def evaluate(model, loader, device, config, logger, epoch, output_dir):
    model.eval()
    scfg = config['sampling']
    all_metrics = []
    vis_cnt = 0
    max_vis = config['training'].get('vis_samples', 3)
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
# main
# ============================================================

def main():
    cfg_path = os.path.join(SYMFOLD_ROOT, 'config', 'train_config.json')
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    config = load_config(cfg_path)
    logger = setup_logging(config)

    try:
        os.setsid()
    except Exception:
        pass

    log_dir = os.path.join(SYMFOLD_ROOT, config['paths']['log_dir'])
    heartbeat_path = os.path.join(log_dir, f'{config["task_name"]}.heartbeat')
    install_signal_handlers(logger, heartbeat_path)
    write_heartbeat(heartbeat_path, {'event': 'start', 'time': time.asctime(),
                                      'pid': os.getpid()})

    logger.info('=' * 70)
    logger.info(f'SymFold Training: {config["task_name"]}')
    logger.info(f'PID={os.getpid()} PGID={os.getpgrp()} py={sys.version.split()[0]}')
    logger.info(f'Torch={torch.__version__} CUDA={torch.version.cuda}')
    logger.info(f'config:\n{json.dumps(config, indent=2)}')
    logger.info('=' * 70)

    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark = True
    # ★ 关键修复: 禁用 TF32 避免 PyTorch 2.4.1 + H20 (Hopper) cuBLAS SIGFPE bug
    # dmesg 中反复出现: traps: python[xxx] trap divide error in libcublasLt.so.12
    # 这是 cuBLAS 内部在 TF32 模式下触发的除零异常，导致进程被 SIGFPE 杀死
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

    logger.info('[Init] building model on CPU...')
    t0 = time.time()
    mc = config['model']
    model = SymFoldModel(
        hidden_dim=mc['hidden_dim'], num_heads=mc['num_heads'],
        dim_head=mc['dim_head'], num_layers=mc['num_layers'],
        patch_size=mc['patch_size'], cond_dim=mc['cond_dim'],
        max_len=mc['max_len'], dp_rate=mc['dp_rate'],
        rho_0=mc['rho_0'], pos_weight_scale=mc['pos_weight_scale'],
        u_ckpt=mc['u_conditioner_ckpt'],
        num_families=mc.get('num_families', 0),
    )
    logger.info(f'[Init] built {time.time()-t0:.1f}s')
    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[Init] params total={total:,} trainable={train_p:,}')

    alphabet = model.get_alphabet()
    train_loader, val_loader = build_dataloaders(config, alphabet, logger)

    logger.info(f'[Init] move to {device}')
    model.to(device)
    logger.info(f'[Init] gpu_alloc={torch.cuda.memory_allocated(device)/1024/1024:.0f}MB')

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training']['lr'])

    # ---- resume ----
    start_epoch = 0
    best_f1 = 0.0
    last_ckpt = os.path.join(model_dir, 'last.pt')
    if config['training'].get('auto_resume', True) and os.path.isfile(last_ckpt):
        try:
            ck = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            start_epoch = ck.get('epoch', 0) + 1
            best_f1 = ck.get('best_f1', 0.0)
            logger.info(f'[Resume] e={start_epoch} best_f1={best_f1}')
        except Exception as e:
            logger.warning(f'[Resume] failed: {e}, from scratch')

    tcfg = config['training']
    eval_every = tcfg.get('eval_every', 5)
    save_every = tcfg.get('save_every', 5)
    patience = tcfg.get('patience', 10)
    patience_cnt = 0
    global_state = {'epoch': start_epoch, 'batch_idx': -1,
                     'last_loss': 0, 'avg_loss': 0}

    def warmup(e):
        if e < tcfg['warmup_epochs']:
            return tcfg['lr'] * (e + 1) / tcfg['warmup_epochs']
        return tcfg['lr']

    try:
        for epoch in range(start_epoch, tcfg['epochs']):
            lr = warmup(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            logger.info(f'[LR] e{epoch}: {lr:.6g}')

            avg = train_one_epoch(model, train_loader, optimizer, device, epoch,
                                   config, logger, heartbeat_path, model_dir,
                                   global_state)

            torch.save({
                'epoch': epoch, 'batch': -1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'avg_loss': avg, 'best_f1': best_f1,
                'config': config,
            }, last_ckpt)

            if (epoch + 1) % eval_every == 0:
                res = evaluate(model, val_loader, device, config, logger,
                                epoch, output_dir)
                f1 = res.get('f1', 0)
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
                        break

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
