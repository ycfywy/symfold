#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取可视化样本的详细信息，生成综合报告
"""
import sys, os, re, json, time
import numpy as np

SYMFOLD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SYMFOLD_ROOT)
sys.path.insert(0, os.path.dirname(SYMFOLD_ROOT))

from datasets.data_generator import _CompatUnpickler


def extract_eval_metrics(log_path):
    """从训练日志中提取所有 eval 结果"""
    eval_results = []
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(
                r'\[Eval\] === e(\d+): F1=([\d.]+) P=([\d.]+) R=([\d.]+) MCC=([\d.]+) N=(\d+) time=([\d.]+)s',
                line)
            if m:
                eval_results.append({
                    'epoch': int(m.group(1)),
                    'F1': float(m.group(2)),
                    'Precision': float(m.group(3)),
                    'Recall': float(m.group(4)),
                    'MCC': float(m.group(5)),
                    'N': int(m.group(6)),
                })
    return eval_results


def extract_sample_info(data_path, target_names):
    """从验证集中提取目标样本的详细信息"""
    with open(data_path, 'rb') as f:
        data = _CompatUnpickler(f).load()
    
    results = []
    for item in data:
        name = str(getattr(item, 'name', ''))
        if name in target_names:
            L = item.length
            seq = getattr(item, 'seq_raw', '')
            pairs_raw = getattr(item, 'pairs', [])
            
            # deduplicate pairs (keep i < j)
            unique_pairs = []
            for p in pairs_raw:
                i, j = int(p[0]), int(p[1])
                if i < j:
                    unique_pairs.append((i, j))
            
            num_pairs = len(unique_pairs)
            paired_bases = num_pairs * 2
            unpaired_bases = L - paired_bases
            
            # nucleotide composition
            comp = {}
            for c in seq.upper():
                comp[c] = comp.get(c, 0) + 1
            
            # pair types
            pair_types = {}
            for (i, j) in unique_pairs:
                a, b = seq[i].upper(), seq[j].upper()
                key = f'{min(a,b)}-{max(a,b)}'
                pair_types[key] = pair_types.get(key, 0) + 1
            
            info = {
                'name': name,
                'length': L,
                'sequence': seq,
                'num_base_pairs': num_pairs,
                'paired_bases': paired_bases,
                'unpaired_bases': unpaired_bases,
                'unpaired_pct': round(100 * unpaired_bases / L, 1),
                'pairing_density': round(paired_bases * 2 / (L * L), 6),
                'nucleotide_composition': comp,
                'pair_types': pair_types,
                'base_pairs_1indexed': [(p[0]+1, p[1]+1) for p in unique_pairs],
            }
            results.append(info)
            
            if len(results) == len(target_names):
                break
    
    results.sort(key=lambda x: x['name'])
    return results


def find_vis_samples(output_dir):
    """扫描可视化输出目录，找出所有被可视化的样本"""
    import glob
    files = glob.glob(os.path.join(output_dir, 'vis_e*_*.png'))
    names = set()
    epochs = set()
    for f in files:
        bn = os.path.basename(f)
        m = re.match(r'vis_e(\d+)_(.+)\.png', bn)
        if m:
            epochs.add(int(m.group(1)))
            names.add(m.group(2))
    return sorted(names), sorted(epochs)


def generate_report(output_dir, log_path, data_path):
    """生成综合报告"""
    # 1. 找出可视化的样本
    sample_names, vis_epochs = find_vis_samples(output_dir)
    print(f'发现 {len(sample_names)} 个可视化样本: {sample_names}')
    print(f'可视化的 epochs: {vis_epochs}')
    
    # 2. 提取 eval metrics
    eval_metrics = extract_eval_metrics(log_path)
    print(f'提取到 {len(eval_metrics)} 次 eval 结果')
    
    # 3. 提取样本详细信息
    sample_infos = extract_sample_info(data_path, sample_names)
    print(f'成功提取 {len(sample_infos)} 个样本的详细信息')
    
    # 4. 生成报告
    report = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_vis_samples': len(sample_names),
            'total_eval_epochs': len(eval_metrics),
            'vis_epochs': vis_epochs,
        },
        'eval_history': eval_metrics,
        'samples': [],
    }
    
    for info in sample_infos:
        report['samples'].append(info)
    
    # 保存
    report_path = os.path.join(output_dir, 'vis_samples_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n报告已保存: {report_path}')
    
    # 生成可读的 markdown 报告
    md_path = os.path.join(output_dir, 'vis_samples_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# SymFold 可视化样本分析报告\n\n')
        f.write(f'**生成时间**: {report["generated_at"]}\n\n')
        
        f.write('## 1. 训练 Eval 历史\n\n')
        f.write('| Epoch | F1 | Precision | Recall | MCC | N |\n')
        f.write('|-------|-----|-----------|--------|-----|----|\n')
        for e in eval_metrics:
            f.write(f'| {e["epoch"]} | {e["F1"]:.4f} | {e["Precision"]:.4f} | {e["Recall"]:.4f} | {e["MCC"]:.4f} | {e["N"]} |\n')
        
        if eval_metrics:
            best = max(eval_metrics, key=lambda x: x['F1'])
            f.write(f'\n**最佳 F1**: {best["F1"]:.4f} (epoch {best["epoch"]})\n\n')
        
        f.write('---\n\n')
        f.write('## 2. 可视化样本详情\n\n')
        
        for idx, s in enumerate(report['samples'], 1):
            f.write(f'### 样本 {idx}: {s["name"]}\n\n')
            f.write(f'| 属性 | 值 |\n')
            f.write(f'|------|----|\n')
            f.write(f'| **序列长度** | {s["length"]} nt |\n')
            f.write(f'| **碱基对数** | {s["num_base_pairs"]} 对 |\n')
            f.write(f'| **配对碱基** | {s["paired_bases"]} / {s["length"]} ({100-s["unpaired_pct"]:.1f}%) |\n')
            f.write(f'| **未配对碱基** | {s["unpaired_bases"]} / {s["length"]} ({s["unpaired_pct"]:.1f}%) |\n')
            f.write(f'| **配对密度** | {s["pairing_density"]:.6f} |\n')
            f.write(f'| **碱基组成** | {s["nucleotide_composition"]} |\n')
            f.write(f'| **配对类型** | {s["pair_types"]} |\n')
            f.write(f'\n**序列**:\n```\n{s["sequence"]}\n```\n\n')
            
            f.write(f'**碱基配对列表** (1-indexed):\n')
            for p in s['base_pairs_1indexed']:
                a = s['sequence'][p[0]-1].upper()
                b = s['sequence'][p[1]-1].upper()
                f.write(f'- {p[0]}({a}) -- {p[1]}({b})\n')
            
            f.write(f'\n**可视化图片** (按 epoch 排列):\n')
            for ep in vis_epochs:
                img = f'vis_e{ep}_{s["name"]}.png'
                f.write(f'- epoch {ep}: `{img}`\n')
            f.write('\n---\n\n')
        
        f.write('## 3. RNA 二级结构说明\n\n')
        f.write('RNA 二级结构天然就是**稀疏**的，大部分碱基处于未配对状态，这是正常现象：\n\n')
        f.write('- **配对区域（stems）**: 连续的 Watson-Crick 配对 (A-U, G-C) 或 G-U wobble 配对\n')
        f.write('- **未配对区域**: loops、bulges、internal loops、single-stranded regions\n')
        f.write('- 平均配对率约 0.5%（contact map 中 99.5% 为 0），这与模型先验 ρ₀=0.005 一致\n\n')
        f.write('**如何阅读可视化图片**:\n')
        f.write('- **左图 (GT)**: Ground Truth contact map, 蓝色表示配对位置\n')
        f.write('- **中图 (Pred)**: 模型预测的 contact map, 橙色表示预测配对\n')
        f.write('- **右图 (Overlay)**: 叠加对比\n')
        f.write('  - 🟢 绿色 = TP (True Positive, 正确预测的配对)\n')
        f.write('  - 🔴 红色 = FN (False Negative, 漏掉的配对)\n')
        f.write('  - 🔵 蓝色 = FP (False Positive, 错误预测的配对)\n')
    
    print(f'Markdown 报告已保存: {md_path}')
    
    return report


if __name__ == '__main__':
    output_dir = os.path.join(SYMFOLD_ROOT, 'output', '260514-full-train-symfold')
    log_path = os.path.join(SYMFOLD_ROOT, 'logs', '260514-full-train-symfold.log')
    data_path = os.path.join(os.path.dirname(SYMFOLD_ROOT), 'data', 'bpRNA', 'VL0.cPickle')
    
    report = generate_report(output_dir, log_path, data_path)
