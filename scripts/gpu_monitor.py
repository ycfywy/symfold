#!/usr/bin/env python3
"""
GPU 实时监控脚本 - 后台运行，持续采集 GPU 数据并绘制曲线

用法:
    cd /root/aigame/dannyyan/RNADiffFold/symfold
    nohup python scripts/gpu_monitor.py --output output/260519-132200-v2-fresh --interval 10 &

参数:
    --output: 输出目录 (默认: output/260519-132200-v2-fresh)
    --interval: 采样间隔秒数 (默认: 10)
    --plot_every: 每隔多少个采样点绘图一次 (默认: 30, 即每5分钟绘一次图)
"""

import subprocess
import time
import json
import os
import sys
import argparse
import signal
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MinuteLocator


def parse_args():
    parser = argparse.ArgumentParser(description='GPU 实时监控')
    parser.add_argument('--output', type=str, 
                        default='output/260519-132200-v2-fresh',
                        help='输出目录')
    parser.add_argument('--interval', type=int, default=10,
                        help='采样间隔(秒)')
    parser.add_argument('--plot_every', type=int, default=30,
                        help='每隔多少个采样点绘图一次')
    return parser.parse_args()


def sample_gpu():
    """采集一次 GPU 数据"""
    try:
        r = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = r.stdout.strip().split(', ')
        if len(parts) >= 5:
            return {
                'timestamp': datetime.now().isoformat(),
                'unix_ts': time.time(),
                'mem_used_mb': int(parts[0]),
                'mem_total_mb': int(parts[1]),
                'gpu_util': int(parts[2]),
                'temp_c': int(parts[3]),
                'power_w': float(parts[4]),
            }
    except Exception as e:
        print(f'[GPU Monitor] Sample error: {e}', file=sys.stderr)
    return None


def plot_gpu_data(records, output_dir):
    """绘制 GPU 监控曲线"""
    if len(records) < 2:
        return
    
    try:
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in records]
        mem = [r['mem_used_mb'] / 1024 for r in records]  # GB
        mem_total = records[0]['mem_total_mb'] / 1024
        util = [r['gpu_util'] for r in records]
        temp = [r['temp_c'] for r in records]
        power = [r['power_w'] for r in records]
        
        duration_min = (records[-1]['unix_ts'] - records[0]['unix_ts']) / 60
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            f'GPU Monitor — H20 96GB | Samples: {len(records)} | '
            f'Duration: {duration_min:.1f} min | '
            f'Updated: {datetime.now().strftime("%H:%M:%S")}',
            fontsize=12, fontweight='bold'
        )
        
        # 显存
        ax = axes[0, 0]
        ax.plot(timestamps, mem, 'b-', linewidth=1.2, alpha=0.8)
        ax.axhline(mem_total, color='r', linestyle='--', alpha=0.4, label=f'Total {mem_total:.0f}GB')
        ax.fill_between(timestamps, mem, alpha=0.15, color='blue')
        ax.set_ylabel('Memory (GB)')
        ax.set_title(f'GPU Memory (peak={max(mem):.1f}GB, avg={np.mean(mem):.1f}GB, now={mem[-1]:.1f}GB)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, mem_total * 1.05)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # 利用率
        ax = axes[0, 1]
        ax.plot(timestamps, util, 'g-', linewidth=1.2, alpha=0.8)
        ax.fill_between(timestamps, util, alpha=0.15, color='green')
        ax.set_ylabel('Utilization (%)')
        ax.set_title(f'GPU Utilization (avg={np.mean(util):.0f}%, now={util[-1]}%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # 温度
        ax = axes[1, 0]
        ax.plot(timestamps, temp, 'orange', linewidth=1.2, alpha=0.8)
        ax.fill_between(timestamps, temp, alpha=0.1, color='orange')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Temperature (max={max(temp)}°C, now={temp[-1]}°C)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # 功耗
        ax = axes[1, 1]
        ax.plot(timestamps, power, 'm-', linewidth=1.2, alpha=0.8)
        ax.fill_between(timestamps, power, alpha=0.1, color='purple')
        ax.set_ylabel('Power (W)')
        ax.set_title(f'Power Draw (avg={np.mean(power):.0f}W, max={max(power):.0f}W, now={power[-1]:.0f}W)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'gpu_monitor_live.png')
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'[GPU Monitor] Plot error: {e}', file=sys.stderr)


def main():
    args = parse_args()
    output_dir = args.output
    interval = args.interval
    plot_every = args.plot_every
    
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'gpu_monitor_live.json')
    
    # 加载已有数据
    records = []
    if os.path.isfile(json_path):
        try:
            with open(json_path, 'r') as f:
                records = json.load(f)
            print(f'[GPU Monitor] Loaded {len(records)} existing records')
        except Exception:
            records = []
    
    # 信号处理 - 优雅退出
    running = [True]
    def handle_signal(sig, frame):
        print(f'\n[GPU Monitor] Received signal {sig}, saving and exiting...')
        running[0] = False
    
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    print(f'[GPU Monitor] Started. output={output_dir}, interval={interval}s, plot_every={plot_every}')
    print(f'[GPU Monitor] PID={os.getpid()}')
    
    count = 0
    while running[0]:
        sample = sample_gpu()
        if sample:
            records.append(sample)
            count += 1
            
            # 定期保存 JSON
            if count % 5 == 0:
                try:
                    with open(json_path, 'w') as f:
                        json.dump(records, f, indent=2)
                except Exception as e:
                    print(f'[GPU Monitor] Save error: {e}', file=sys.stderr)
            
            # 定期绘图
            if count % plot_every == 0:
                plot_gpu_data(records, output_dir)
                print(f'[GPU Monitor] {datetime.now().strftime("%H:%M:%S")} '
                      f'samples={len(records)} '
                      f'mem={sample["mem_used_mb"]/1024:.1f}GB '
                      f'util={sample["gpu_util"]}% '
                      f'temp={sample["temp_c"]}°C '
                      f'power={sample["power_w"]:.0f}W')
        
        time.sleep(interval)
    
    # 退出前保存
    if records:
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2)
        plot_gpu_data(records, output_dir)
        print(f'[GPU Monitor] Final save: {len(records)} records')
    
    print('[GPU Monitor] Done.')


if __name__ == '__main__':
    main()
