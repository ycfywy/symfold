#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SymFold V3 Eval Case Analysis — 分析表现好/差的样本特征

用法:
    cd /root/aigame/dannyyan/RNADiffFold/symfold
    python scripts/analyze_eval_cases.py output/260520-v3-train/eval_detailed.json

输出:
    - 控制台: 统计摘要
    - output/<task>/case_analysis.json: 详细分析结果
    - output/<task>/case_analysis_plots.png: 可视化
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

def load_eval_results(path):
    with open(path) as f:
        return json.load(f)

def analyze_dataset(ds_name, ds_data):
    """分析单个数据集的好/差 case 特征"""
    samples = ds_data['all_samples_summary']
    if not samples:
        return None
    
    # 基本统计
    f1s = [s['f1'] for s in samples]
    lengths = [s['length'] for s in samples]
    gt_pairs = [s['gt_num_pairs'] for s in samples]
    pred_pairs = [s['pred_num_pairs'] for s in samples]
    precisions = [s['precision'] for s in samples]
    recalls = [s['recall'] for s in samples]
    
    # 计算衍生指标
    pair_rates = [s['gt_num_pairs'] / (s['length'] * (s['length'] - 1) / 2) if s['length'] > 1 else 0 for s in samples]
    pair_per_base = [2 * s['gt_num_pairs'] / s['length'] if s['length'] > 0 else 0 for s in samples]
    pred_ratio = [s['pred_num_pairs'] / max(s['gt_num_pairs'], 1) for s in samples]
    
    # 分组: 好 (F1>=0.8), 中 (0.5<=F1<0.8), 差 (F1<0.5)
    good = [s for s in samples if s['f1'] >= 0.8]
    medium = [s for s in samples if 0.5 <= s['f1'] < 0.8]
    bad = [s for s in samples if s['f1'] < 0.5]
    
    # 极差 (F1=0)
    zero_f1 = [s for s in samples if s['f1'] == 0]
    
    # 完美 (F1=1.0)
    perfect = [s for s in samples if s['f1'] >= 0.999]
    
    def group_stats(group, label):
        if not group:
            return {'label': label, 'count': 0}
        gl = [s['length'] for s in group]
        gp = [s['gt_num_pairs'] for s in group]
        gpb = [2 * s['gt_num_pairs'] / s['length'] for s in group]
        gpr = [s['pred_num_pairs'] / max(s['gt_num_pairs'], 1) for s in group]
        gprec = [s['precision'] for s in group]
        grec = [s['recall'] for s in group]
        return {
            'label': label,
            'count': len(group),
            'pct': f'{100*len(group)/len(samples):.1f}%',
            'length': {'mean': np.mean(gl), 'median': np.median(gl), 'min': min(gl), 'max': max(gl)},
            'gt_pairs': {'mean': np.mean(gp), 'median': np.median(gp)},
            'pair_per_base': {'mean': np.mean(gpb), 'median': np.median(gpb)},
            'pred_ratio': {'mean': np.mean(gpr), 'median': np.median(gpr), 'desc': 'pred_pairs/gt_pairs'},
            'precision': {'mean': np.mean(gprec), 'median': np.median(gprec)},
            'recall': {'mean': np.mean(grec), 'median': np.median(grec)},
        }
    
    # 按长度分 bin 分析 F1
    length_bins = [(0, 80), (80, 160), (160, 240), (240, 320), (320, 480), (480, 640)]
    length_analysis = []
    for lo, hi in length_bins:
        bin_samples = [s for s in samples if lo <= s['length'] < hi]
        if bin_samples:
            bin_f1s = [s['f1'] for s in bin_samples]
            length_analysis.append({
                'range': f'[{lo},{hi})',
                'count': len(bin_samples),
                'f1_mean': np.mean(bin_f1s),
                'f1_std': np.std(bin_f1s),
                'f1_median': np.median(bin_f1s),
                'bad_rate': sum(1 for f in bin_f1s if f < 0.5) / len(bin_f1s),
            })
    
    # pair_per_base 分 bin 分析
    ppb_bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    ppb_analysis = []
    for lo, hi in ppb_bins:
        bin_samples = [s for s in samples if lo <= 2*s['gt_num_pairs']/max(s['length'],1) < hi]
        if bin_samples:
            bin_f1s = [s['f1'] for s in bin_samples]
            ppb_analysis.append({
                'range': f'[{lo},{hi})',
                'count': len(bin_samples),
                'f1_mean': np.mean(bin_f1s),
                'bad_rate': sum(1 for f in bin_f1s if f < 0.5) / len(bin_f1s),
            })
    
    # 相关性分析
    f1_arr = np.array(f1s)
    len_arr = np.array(lengths)
    gp_arr = np.array(gt_pairs)
    ppb_arr = np.array(pair_per_base)
    pr_arr = np.array(pred_ratio)
    
    correlations = {
        'f1_vs_length': float(np.corrcoef(f1_arr, len_arr)[0, 1]) if len(f1_arr) > 2 else 0,
        'f1_vs_gt_pairs': float(np.corrcoef(f1_arr, gp_arr)[0, 1]) if len(f1_arr) > 2 else 0,
        'f1_vs_pair_per_base': float(np.corrcoef(f1_arr, ppb_arr)[0, 1]) if len(f1_arr) > 2 else 0,
        'f1_vs_pred_ratio': float(np.corrcoef(f1_arr, pr_arr)[0, 1]) if len(f1_arr) > 2 else 0,
    }
    
    # worst 10 samples
    sorted_by_f1 = sorted(samples, key=lambda x: x['f1'])
    worst_10 = sorted_by_f1[:10]
    
    # 过度预测 (pred >> gt) vs 欠预测 (pred << gt)
    over_predict = [s for s in samples if s['pred_num_pairs'] > 1.5 * max(s['gt_num_pairs'], 1)]
    under_predict = [s for s in samples if s['pred_num_pairs'] < 0.5 * s['gt_num_pairs'] and s['gt_num_pairs'] > 3]
    
    return {
        'dataset': ds_name,
        'total_samples': len(samples),
        'overall': {
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'f1_median': float(np.median(f1s)),
            'length_mean': float(np.mean(lengths)),
            'length_std': float(np.std(lengths)),
        },
        'groups': {
            'good_f1_ge_0.8': group_stats(good, 'F1≥0.8'),
            'medium_0.5_to_0.8': group_stats(medium, '0.5≤F1<0.8'),
            'bad_f1_lt_0.5': group_stats(bad, 'F1<0.5'),
            'zero_f1': group_stats(zero_f1, 'F1=0'),
            'perfect_f1': group_stats(perfect, 'F1=1.0'),
        },
        'by_length': length_analysis,
        'by_pair_density': ppb_analysis,
        'correlations': correlations,
        'over_predict': {
            'count': len(over_predict),
            'pct': f'{100*len(over_predict)/len(samples):.1f}%',
            'avg_f1': float(np.mean([s['f1'] for s in over_predict])) if over_predict else 0,
        },
        'under_predict': {
            'count': len(under_predict),
            'pct': f'{100*len(under_predict)/len(samples):.1f}%',
            'avg_f1': float(np.mean([s['f1'] for s in under_predict])) if under_predict else 0,
        },
        'worst_10': worst_10,
    }


def plot_analysis(all_results, output_path):
    """绘制分析图"""
    n_ds = len(all_results)
    fig, axes = plt.subplots(3, n_ds, figsize=(5 * n_ds, 12))
    if n_ds == 1:
        axes = axes.reshape(3, 1)
    
    for col, (ds_name, analysis) in enumerate(all_results.items()):
        if analysis is None:
            continue
        
        # Row 1: F1 distribution histogram
        ax = axes[0, col]
        # 从 worst_10 和 overall 重建数据... 需要原始数据
        # 这里用 by_length 来画 bar
        bins = analysis['by_length']
        if bins:
            names = [b['range'] for b in bins]
            f1s = [b['f1_mean'] for b in bins]
            counts = [b['count'] for b in bins]
            bad_rates = [b['bad_rate'] for b in bins]
            
            x = range(len(names))
            ax.bar(x, f1s, color='steelblue', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, fontsize=8)
            ax.set_ylabel('Mean F1')
            ax.set_title(f'{ds_name}\nF1 by Length')
            ax.set_ylim(0, 1)
            ax.axhline(y=analysis['overall']['f1_mean'], color='red', linestyle='--', alpha=0.5)
        
        # Row 2: Pair density vs F1
        ax = axes[1, col]
        ppb_bins = analysis['by_pair_density']
        if ppb_bins:
            names = [b['range'] for b in ppb_bins]
            f1s = [b['f1_mean'] for b in ppb_bins]
            x = range(len(names))
            ax.bar(x, f1s, color='forestgreen', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, fontsize=8)
            ax.set_ylabel('Mean F1')
            ax.set_title(f'{ds_name}\nF1 by Pair Density')
            ax.set_ylim(0, 1)
        
        # Row 3: Good/Medium/Bad distribution pie
        ax = axes[2, col]
        groups = analysis['groups']
        sizes = [groups['good_f1_ge_0.8']['count'],
                 groups['medium_0.5_to_0.8']['count'],
                 groups['bad_f1_lt_0.5']['count']]
        labels = [f'Good≥0.8\n({sizes[0]})', f'Medium\n({sizes[1]})', f'Bad<0.5\n({sizes[2]})']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{ds_name}\nQuality Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[Plot] saved to {output_path}')


def print_summary(all_results):
    """控制台输出分析摘要"""
    print('=' * 80)
    print('V3 EVAL CASE ANALYSIS SUMMARY')
    print('=' * 80)
    
    for ds_name, analysis in all_results.items():
        if analysis is None:
            continue
        print(f'\n{"─"*70}')
        print(f'  {ds_name} (N={analysis["total_samples"]})')
        print(f'{"─"*70}')
        
        ov = analysis['overall']
        print(f'  Overall: F1={ov["f1_mean"]:.4f}±{ov["f1_std"]:.4f} (median={ov["f1_median"]:.4f})')
        print(f'  Length: {ov["length_mean"]:.1f}±{ov["length_std"]:.1f}')
        
        # Groups
        g = analysis['groups']
        print(f'\n  Quality Groups:')
        print(f'    Good (F1≥0.8):  {g["good_f1_ge_0.8"]["count"]:5d} ({g["good_f1_ge_0.8"].get("pct","0%")})')
        print(f'    Medium:         {g["medium_0.5_to_0.8"]["count"]:5d} ({g["medium_0.5_to_0.8"].get("pct","0%")})')
        print(f'    Bad (F1<0.5):   {g["bad_f1_lt_0.5"]["count"]:5d} ({g["bad_f1_lt_0.5"].get("pct","0%")})')
        print(f'    Perfect (F1=1): {g["perfect_f1"]["count"]:5d} ({g["perfect_f1"].get("pct","0%")})')
        print(f'    Zero (F1=0):    {g["zero_f1"]["count"]:5d} ({g["zero_f1"].get("pct","0%")})')
        
        # Bad vs Good comparison
        if g['bad_f1_lt_0.5']['count'] > 0 and g['good_f1_ge_0.8']['count'] > 0:
            bad = g['bad_f1_lt_0.5']
            good = g['good_f1_ge_0.8']
            print(f'\n  Bad vs Good Comparison:')
            print(f'    {"":20s} {"Bad (F1<0.5)":>15s} {"Good (F1≥0.8)":>15s}')
            print(f'    {"Length mean":20s} {bad["length"]["mean"]:>15.1f} {good["length"]["mean"]:>15.1f}')
            print(f'    {"Pair/base mean":20s} {bad["pair_per_base"]["mean"]:>15.3f} {good["pair_per_base"]["mean"]:>15.3f}')
            print(f'    {"Pred/GT ratio":20s} {bad["pred_ratio"]["mean"]:>15.3f} {good["pred_ratio"]["mean"]:>15.3f}')
            print(f'    {"Precision":20s} {bad["precision"]["mean"]:>15.3f} {good["precision"]["mean"]:>15.3f}')
            print(f'    {"Recall":20s} {bad["recall"]["mean"]:>15.3f} {good["recall"]["mean"]:>15.3f}')
        
        # Correlations
        c = analysis['correlations']
        print(f'\n  Correlations with F1:')
        print(f'    Length:       r={c["f1_vs_length"]:+.3f}')
        print(f'    GT pairs:    r={c["f1_vs_gt_pairs"]:+.3f}')
        print(f'    Pair/base:   r={c["f1_vs_pair_per_base"]:+.3f}')
        print(f'    Pred ratio:  r={c["f1_vs_pred_ratio"]:+.3f}')
        
        # By length
        print(f'\n  F1 by Sequence Length:')
        for b in analysis['by_length']:
            bar = '█' * int(b['f1_mean'] * 20)
            print(f'    {b["range"]:>10s}: F1={b["f1_mean"]:.3f} (N={b["count"]:4d}, bad_rate={100*b["bad_rate"]:.1f}%) {bar}')
        
        # Over/Under predict
        op = analysis['over_predict']
        up = analysis['under_predict']
        print(f'\n  Prediction Behavior:')
        print(f'    Over-predict (pred>1.5×gt): {op["count"]:4d} ({op["pct"]}), avg F1={op["avg_f1"]:.3f}')
        print(f'    Under-predict (pred<0.5×gt): {up["count"]:4d} ({up["pct"]}), avg F1={up["avg_f1"]:.3f}')
        
        # Worst 10
        print(f'\n  Worst 10 Samples:')
        for s in analysis['worst_10']:
            ppb = 2 * s['gt_num_pairs'] / max(s['length'], 1)
            print(f'    {s["name"][:40]:40s} L={s["length"]:3d} F1={s["f1"]:.3f} P={s["precision"]:.3f} R={s["recall"]:.3f} '
                  f'gt={s["gt_num_pairs"]:3d} pred={s["pred_num_pairs"]:3d} ppb={ppb:.2f}')


def main():
    if len(sys.argv) < 2:
        eval_path = 'output/260520-v3-train/eval_detailed.json'
    else:
        eval_path = sys.argv[1]
    
    if not os.path.isfile(eval_path):
        print(f'ERROR: {eval_path} not found')
        sys.exit(1)
    
    data = load_eval_results(eval_path)
    results = data['results']
    
    all_results = {}
    for ds_name, ds_data in results.items():
        if 'all_samples_summary' in ds_data and ds_data['all_samples_summary']:
            all_results[ds_name] = analyze_dataset(ds_name, ds_data)
        else:
            all_results[ds_name] = None
    
    # Print summary
    print_summary(all_results)
    
    # Save JSON
    output_dir = os.path.dirname(eval_path)
    json_path = os.path.join(output_dir, 'case_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f'\n[Save] Analysis JSON: {json_path}')
    
    # Plot
    plot_path = os.path.join(output_dir, 'case_analysis_plots.png')
    plot_analysis(all_results, plot_path)
    
    print('\nDone.')


if __name__ == '__main__':
    main()
