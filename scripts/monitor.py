#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SymFold 训练实时监控脚本

功能:
  1. 解析训练日志，提取 batch loss / epoch avg_loss / eval F1,P,R,MCC
  2. 绘制多面板训练曲线 (batch loss + epoch loss + eval metrics)
  3. 定期自动刷新图片，不影响训练进程
  4. 输出到 output/<task>/training_curves.png

用法:
  # 一次性生成当前曲线
  python scripts/monitor.py logs/260514-full-train-symfold.log

  # 持续监控 (每 60s 刷新)
  python scripts/monitor.py logs/260514-full-train-symfold.log --watch 60
"""
import re
import sys
import os
import time
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime


def parse_log(log_path):
    """从日志文件中提取训练信息"""
    batch_losses = []       # (global_step, loss, epoch, batch, L, bs)
    epoch_summaries = []    # (epoch, avg_loss, success, total, time_s)
    eval_results = []       # (epoch, f1, prec, rec, mcc, N, time_s)
    lr_changes = []         # (epoch, lr)
    best_saves = []         # (epoch, f1)
    patience_info = []      # (epoch, cnt, max_patience)
    oom_events = []         # (epoch, batch, L)

    global_step = 0

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Batch loss: [Train] e0 b20/1758 L=480 bs=3 loss=1.238879 t=0.141s
            m = re.search(
                r'\[Train\] e(\d+) b(\d+)/(\d+) L=(\d+) bs=(\d+) loss=([\d.]+) t=([\d.]+)s',
                line)
            if m:
                epoch = int(m.group(1))
                batch = int(m.group(2))
                total = int(m.group(3))
                L = int(m.group(4))
                bs = int(m.group(5))
                loss = float(m.group(6))
                t = float(m.group(7))
                global_step = epoch * total + batch
                batch_losses.append({
                    'step': global_step, 'epoch': epoch, 'batch': batch,
                    'total': total, 'L': L, 'bs': bs, 'loss': loss, 'time': t,
                })
                continue

            # Epoch summary: [Train] === e0 done: avg_loss=0.069269 success=4845/4845 time=560.0s ===
            m = re.search(
                r'\[Train\] === e(\d+) done: avg_loss=([\d.]+) success=(\d+)/(\d+) time=([\d.]+)s',
                line)
            if m:
                epoch_summaries.append({
                    'epoch': int(m.group(1)),
                    'avg_loss': float(m.group(2)),
                    'success': int(m.group(3)),
                    'total': int(m.group(4)),
                    'time_s': float(m.group(5)),
                })
                continue

            # Eval: [Eval] === e1: F1=0.1234 P=0.1500 R=0.1100 MCC=0.0800 N=1299 time=45.2s ===
            m = re.search(
                r'\[Eval\] === e(\d+): F1=([\d.]+) P=([\d.]+) R=([\d.]+) MCC=([\d.]+) N=(\d+) time=([\d.]+)s',
                line)
            if m:
                eval_results.append({
                    'epoch': int(m.group(1)),
                    'f1': float(m.group(2)),
                    'precision': float(m.group(3)),
                    'recall': float(m.group(4)),
                    'mcc': float(m.group(5)),
                    'N': int(m.group(6)),
                    'time_s': float(m.group(7)),
                })
                continue

            # LR: [LR] e0: 0.0001
            m = re.search(r'\[LR\] e(\d+): ([\d.e-]+)', line)
            if m:
                lr_changes.append({
                    'epoch': int(m.group(1)),
                    'lr': float(m.group(2)),
                })
                continue

            # Best save: [Save] new best F1=0.xxxx
            m = re.search(r'\[Save\] new best F1=([\d.]+)', line)
            if m:
                f1 = float(m.group(1))
                ep = eval_results[-1]['epoch'] if eval_results else 0
                best_saves.append({'epoch': ep, 'f1': f1})
                continue

            # Patience: [Eval] no improve 3/15
            m = re.search(r'\[Eval\] no improve (\d+)/(\d+)', line)
            if m:
                cnt = int(m.group(1))
                mx = int(m.group(2))
                ep = eval_results[-1]['epoch'] if eval_results else 0
                patience_info.append({'epoch': ep, 'cnt': cnt, 'max': mx})
                continue

            # OOM: [OOM] e0 b123 L=640 skip
            m = re.search(r'\[OOM\] e(\d+) b(\d+) L=(\d+)', line)
            if m:
                oom_events.append({
                    'epoch': int(m.group(1)),
                    'batch': int(m.group(2)),
                    'L': int(m.group(3)),
                })
                continue

    return {
        'batch_losses': batch_losses,
        'epoch_summaries': epoch_summaries,
        'eval_results': eval_results,
        'lr_changes': lr_changes,
        'best_saves': best_saves,
        'patience_info': patience_info,
        'oom_events': oom_events,
    }


def moving_average(vals, window=50):
    if len(vals) < window:
        window = max(1, len(vals) // 3)
    if window < 1:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode='valid')


def plot_curves(data, save_path, task_name=''):
    bl = data['batch_losses']
    es = data['epoch_summaries']
    ev = data['eval_results']
    lr = data['lr_changes']
    bs = data['best_saves']
    pi = data['patience_info']
    oom = data['oom_events']

    has_eval = len(ev) > 0
    n_panels = 4 if has_eval else 3

    fig = plt.figure(figsize=(20, 5 * n_panels))
    gs = gridspec.GridSpec(n_panels, 1, hspace=0.35)

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title = f'SymFold Training Monitor'
    if task_name:
        title += f' — {task_name}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # ── Panel 1: Batch Loss (log scale + moving avg) ──
    ax1 = fig.add_subplot(gs[0])
    if bl:
        steps = [b['step'] for b in bl]
        losses = [b['loss'] for b in bl]
        ax1.scatter(steps, losses, s=1, alpha=0.15, c='steelblue', label='batch loss')
        if len(losses) > 10:
            ma = moving_average(losses, window=min(100, len(losses)//5))
            offset = len(losses) - len(ma)
            ax1.plot(steps[offset:], ma, c='darkblue', lw=2, label=f'MA-{min(100, len(losses)//5)}')
        ax1.set_yscale('log')
        ax1.set_xlabel('Global Step')
        ax1.set_ylabel('Loss (log)')
        ax1.set_title(f'Batch Loss  |  {len(bl)} steps logged')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 标注 epoch 边界
        for e in es:
            ep = e['epoch']
            step_end = (ep + 1) * bl[0]['total'] if bl else 0
            ax1.axvline(step_end, color='gray', ls='--', alpha=0.3, lw=0.8)

    # ── Panel 2: Epoch Avg Loss + LR ──
    ax2 = fig.add_subplot(gs[1])
    if es:
        epochs = [e['epoch'] for e in es]
        avg_losses = [e['avg_loss'] for e in es]
        ax2.plot(epochs, avg_losses, 'o-', c='coral', lw=2, markersize=6, label='avg loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Avg Loss', color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax2.set_title(f'Epoch Summary  |  {len(es)} epochs done')
        ax2.grid(True, alpha=0.3)

        # LR on secondary axis
        if lr:
            ax2r = ax2.twinx()
            lr_epochs = [l['epoch'] for l in lr]
            lr_vals = [l['lr'] for l in lr]
            ax2r.plot(lr_epochs, lr_vals, 's--', c='green', alpha=0.7, markersize=4, label='lr')
            ax2r.set_ylabel('Learning Rate', color='green')
            ax2r.tick_params(axis='y', labelcolor='green')

        ax2.legend(loc='upper right')
    else:
        # 如果还没完成一个 epoch，从 batch loss 估算
        if bl:
            curr_epoch = bl[-1]['epoch']
            curr_batch = bl[-1]['batch']
            curr_total = bl[-1]['total']
            progress = curr_batch / curr_total * 100
            epoch_losses = [b['loss'] for b in bl if b['epoch'] == curr_epoch]
            running_avg = np.mean(epoch_losses) if epoch_losses else 0
            ax2.text(0.5, 0.5,
                     f'Epoch {curr_epoch} in progress: {curr_batch}/{curr_total} '
                     f'({progress:.1f}%)\nRunning avg loss: {running_avg:.6f}',
                     ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Epoch Summary (no complete epoch yet)')

    # ── Panel 3: Eval Metrics (F1, Precision, Recall, MCC) ──
    ax3 = fig.add_subplot(gs[2])
    if has_eval:
        ep_eval = [e['epoch'] for e in ev]
        f1s = [e['f1'] for e in ev]
        precs = [e['precision'] for e in ev]
        recs = [e['recall'] for e in ev]
        mccs = [e['mcc'] for e in ev]

        ax3.plot(ep_eval, f1s, 'o-', c='#e74c3c', lw=2.5, markersize=8, label='F1', zorder=5)
        ax3.plot(ep_eval, precs, 's--', c='#3498db', lw=1.5, markersize=5, label='Precision')
        ax3.plot(ep_eval, recs, '^--', c='#2ecc71', lw=1.5, markersize=5, label='Recall')
        ax3.plot(ep_eval, mccs, 'D--', c='#9b59b6', lw=1.5, markersize=5, label='MCC')

        # 标注 best
        if bs:
            for b in bs:
                ax3.annotate(f'best={b["f1"]:.4f}',
                             xy=(b['epoch'], b['f1']),
                             fontsize=10, fontweight='bold', color='red',
                             xytext=(5, 10), textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', color='red'))

        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title(f'Validation Metrics  |  Best F1={max(f1s):.4f}' if f1s else 'Validation')
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(loc='lower right', ncol=4)
        ax3.grid(True, alpha=0.3)

        # Patience bar
        if pi:
            ax3r = ax3.twinx()
            p_ep = [p['epoch'] for p in pi]
            p_cnt = [p['cnt'] for p in pi]
            p_max = pi[0]['max'] if pi else 15
            ax3r.bar(p_ep, p_cnt, alpha=0.15, color='gray', width=0.8, label='patience')
            ax3r.axhline(p_max, color='gray', ls=':', alpha=0.5)
            ax3r.set_ylabel('Patience cnt', color='gray')
            ax3r.set_ylim(0, p_max + 2)
    else:
        ax3.text(0.5, 0.5, 'No eval results yet\n(eval runs every 2 epochs)',
                 ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.set_title('Validation Metrics (waiting...)')

    # ── Panel 4: Training speed & info ──
    if n_panels >= 4:
        ax4 = fig.add_subplot(gs[3])
    else:
        ax4 = fig.add_subplot(gs[2]) if not has_eval else fig.add_subplot(gs[n_panels - 1])

    # Speed per batch
    if bl:
        steps_plot = [b['step'] for b in bl]
        times_plot = [b['time'] for b in bl]
        bs_plot = [b['bs'] for b in bl]
        throughput = [b['bs'] / max(b['time'], 0.001) for b in bl]

        ax4.scatter(steps_plot, throughput, s=2, alpha=0.3, c='teal')
        if len(throughput) > 10:
            ma_tp = moving_average(throughput, window=min(50, len(throughput)//5))
            offset = len(throughput) - len(ma_tp)
            ax4.plot(steps_plot[offset:], ma_tp, c='darkcyan', lw=2)
        ax4.set_xlabel('Global Step')
        ax4.set_ylabel('Throughput (samples/s)')
        ax4.set_title('Training Speed')
        ax4.grid(True, alpha=0.3)

        # OOM markers
        if oom:
            for o in oom:
                s = o['epoch'] * bl[0]['total'] + o['batch']
                ax4.axvline(s, color='red', ls='-', alpha=0.5, lw=1)
            ax4.plot([], [], 'r-', label=f'OOM events ({len(oom)})')
            ax4.legend()

    # ── Info text box ──
    info_lines = [f'Updated: {now_str}']
    if bl:
        curr_e = bl[-1]['epoch']
        curr_b = bl[-1]['batch']
        curr_t = bl[-1]['total']
        info_lines.append(f'Current: epoch {curr_e}, batch {curr_b}/{curr_t}')
    if es:
        info_lines.append(f'Completed epochs: {len(es)}')
        info_lines.append(f'Last epoch loss: {es[-1]["avg_loss"]:.6f}')
    if ev:
        info_lines.append(f'Best val F1: {max(e["f1"] for e in ev):.4f}')
    if oom:
        info_lines.append(f'OOM events: {len(oom)}')

    fig.text(0.01, 0.005, '  |  '.join(info_lines), fontsize=9, color='gray',
             style='italic')

    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description='SymFold Training Monitor')
    parser.add_argument('log_path', help='Path to .log file')
    parser.add_argument('--watch', type=int, default=0,
                        help='Refresh interval in seconds (0=one-shot)')
    parser.add_argument('--out', type=str, default='',
                        help='Output image path (default: auto from log name)')
    args = parser.parse_args()

    log_path = args.log_path
    if not os.path.isabs(log_path):
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            log_path)

    # Auto output path
    if args.out:
        out_path = args.out
    else:
        base = os.path.basename(log_path).replace('.log', '')
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'output', base)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'training_curves.png')

    task_name = os.path.basename(log_path).replace('.log', '')

    def run_once():
        if not os.path.isfile(log_path):
            print(f'[Monitor] Waiting for log file: {log_path}')
            return
        data = parse_log(log_path)
        n_batches = len(data['batch_losses'])
        n_epochs = len(data['epoch_summaries'])
        n_evals = len(data['eval_results'])
        saved = plot_curves(data, out_path, task_name)
        print(f'[Monitor] {datetime.now().strftime("%H:%M:%S")} | '
              f'steps={n_batches} epochs={n_epochs} evals={n_evals} | '
              f'saved → {saved}')

    if args.watch > 0:
        print(f'[Monitor] Watching {log_path} every {args.watch}s ...')
        print(f'[Monitor] Output: {out_path}')
        while True:
            try:
                run_once()
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print('\n[Monitor] stopped')
                break
    else:
        run_once()


if __name__ == '__main__':
    main()
