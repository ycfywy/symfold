# -*- coding: utf-8 -*-
"""
SymFold 评估入口: 在多个测试集上测试 checkpoint.

用法:
    cd /root/aigame/dannyyan/RNADiffFold/symfold
    python eval/eval.py --ckpt model/<task>/best.pt --test_sets bpRNA,ArchiveII,PDB_TS1

    详细模式 (输出逐样本结果 + 序列 + GT/Pred 可视化):
    python eval/eval.py --ckpt model/<task>/best.pt --test_sets PDB_TS1,PDB_TS3 --detailed

    或者用 config 方式:
    python eval/eval.py --config config/eval_config.json
"""
from __future__ import annotations

import sys
import os
import json
import time
import argparse
from functools import partial
from typing import List, Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SYMFOLD_ROOT = os.path.dirname(EVAL_DIR)
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


# 测试集文件映射
TEST_SET_FILES = {
    'bpRNA':        ['data/bpRNA/TS0.cPickle'],
    'RNAStrAlign':  ['data/RNAStrAlign/test.cPickle'],
    'bpRNA-new':    ['data/bpRNA-new/bpRNAnew.cPickle'],
    'ArchiveII':    ['data/ArchiveII/archiveII.cPickle'],
    'PDB_TS1':      ['data/PDB/TS1.cPickle'],
    'PDB_TS2':      ['data/PDB/TS2.cPickle'],
    'PDB_TS3':      ['data/PDB/TS3.cPickle'],
    'PDB_TS_hard':  ['data/PDB/TS_hard.cPickle'],
}


def load_model_from_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck['config']
    mc = cfg['model']
    model = SymFoldModel(
        hidden_dim=mc['hidden_dim'], num_heads=mc['num_heads'],
        dim_head=mc['dim_head'], num_layers=mc['num_layers'],
        patch_size=mc['patch_size'], cond_dim=mc['cond_dim'],
        max_len=mc['max_len'], dp_rate=mc.get('dp_rate', 0.0),
        rho_0=mc['rho_0'], pos_weight_scale=mc['pos_weight_scale'],
        u_ckpt=mc['u_conditioner_ckpt'],
        num_families=mc.get('num_families', 0),
    )
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    return model, cfg


def _contact_to_pairs(contact_map: np.ndarray, length: int):
    """从 contact map 中提取碱基配对列表 [(i,j), ...], 只取上三角"""
    pairs = []
    for i in range(length):
        for j in range(i + 1, length):
            if contact_map[i, j] > 0.5:
                pairs.append((i, j))
    return pairs


def _pairs_to_dot_bracket(pairs, length: int) -> str:
    """将碱基对列表转为 dot-bracket 表示 (仅嵌套部分, 伪结用 [] 表示)"""
    db = ['.'] * length
    # 先按 nested 方式分配
    used = set()
    # 按 (i, j-i) 排序, 先分配短程 pair
    sorted_pairs = sorted(pairs, key=lambda p: (p[0], p[1]))
    for i, j in sorted_pairs:
        if i not in used and j not in used:
            # 检查是否与已有 pair 交叉 (pseudoknot)
            is_nested = True
            for pi, pj in [(x, y) for x, y in pairs if (x, y) != (i, j)]:
                if pi in used or pj in used:
                    continue
                # 交叉: i < pi < j < pj 或 pi < i < pj < j
                if (i < pi < j < pj) or (pi < i < pj < j):
                    pass  # 可能是 pseudoknot, 后续再处理
            db[i] = '('
            db[j] = ')'
            used.add(i)
            used.add(j)
    return ''.join(db)


def _format_sequence_display(seq: str, length: int, max_display: int = 80) -> str:
    """格式化序列显示, 过长时截断"""
    s = seq[:length]
    if len(s) > max_display:
        return s[:max_display - 3] + '...'
    return s


@torch.no_grad()
def eval_on_dataset(model, files: List[str], device,
                    num_steps: int = 20,
                    num_samples_per_input: int = 1,
                    physics_beta: float = 0.0,
                    physics_lambda_pk: float = 0.0,
                    bucket_bs=None,
                    max_set_len: int = 640,
                    log_prefix: str = '',
                    detailed: bool = False):
    """在给定 pickle 文件列表上跑 sample + metric.

    如果 detailed=True, 额外返回逐样本详细信息:
      - name, length, sequence, gt_pairs, pred_pairs, metrics, dot_bracket_gt, dot_bracket_pred
    """
    roots = []
    abs_root = os.path.dirname(SYMFOLD_ROOT)
    for f in files:
        if not os.path.isabs(f):
            f = os.path.join(abs_root, f)
        roots.append(f)

    idx = build_index(roots, verbose=False)
    if not idx:
        print(f'{log_prefix} no samples in {roots}')
        return None

    bs_table = bucket_bs or {80: 8, 160: 4, 240: 2, 320: 1,
                              400: 1, 480: 1, 560: 1, 640: 1}
    sampler = BucketBatchSampler(idx, batch_size_table=bs_table, shuffle=False,
                                  max_set_len=max_set_len, seed=0)
    collate = partial(simple_collate_fn, alphabet=model.get_alphabet())
    loader = DataLoader(SimpleRNADataset(idx), batch_sampler=sampler,
                         collate_fn=collate, num_workers=0)

    all_metrics = []
    per_len_metrics = {}
    sample_details = []  # 逐样本详情
    t0 = time.time()
    for bi, batch in enumerate(loader):
        try:
            contact = batch['contact']
            seq_oh = batch['seq_oh'].to(device)
            seq_enc = batch['seq_enc'].to(device)
            tokens = batch['tokens'].to(device)
            length = batch['length'].to(device)
            set_max_len_val = int(batch['set_max_len'])
            names = batch.get('names', [f'sample_{bi}_{i}' for i in range(contact.shape[0])])
            seq_raws = batch.get('seq_raws', [''] * contact.shape[0])
            data_fcn_2 = get_data_fcn_gpu(seq_oh, set_max_len_val)
            matrix_rep = torch.zeros_like(contact)
            cm = contact_map_masks(length, matrix_rep).to(device)
            pred, _ = model.sample(
                data_fcn_2=data_fcn_2, tokens=tokens,
                contact_masks=cm, set_max_len=set_max_len_val,
                seq_oh=seq_enc,
                num_steps=num_steps,
                num_samples_per_input=num_samples_per_input,
                physics_beta=physics_beta,
                physics_lambda_pk=physics_lambda_pk,
            )
            pred = pred.cpu().float()
            for i in range(contact.shape[0]):
                m = rna_evaluation(pred[i].squeeze(), contact.float()[i].squeeze())
                all_metrics.append(m)
                per_len_metrics.setdefault(set_max_len_val, []).append(m)

                if detailed:
                    seq_len = int(length[i].item())
                    gt_map = contact[i].squeeze().numpy()[:seq_len, :seq_len]
                    pred_map = pred[i].squeeze().numpy()[:seq_len, :seq_len]
                    gt_pairs = _contact_to_pairs(gt_map, seq_len)
                    pred_pairs = _contact_to_pairs(pred_map, seq_len)
                    gt_db = _pairs_to_dot_bracket(gt_pairs, seq_len)
                    pred_db = _pairs_to_dot_bracket(pred_pairs, seq_len)
                    seq_str = seq_raws[i] if i < len(seq_raws) else ''
                    # 计算详细统计
                    gt_set = set(gt_pairs)
                    pred_set = set(pred_pairs)
                    tp_pairs = gt_set & pred_set
                    fp_pairs = pred_set - gt_set
                    fn_pairs = gt_set - pred_set
                    _prec = float(m[1].item()) if hasattr(m[1], 'item') else float(m[1])
                    _rec = float(m[2].item()) if hasattr(m[2], 'item') else float(m[2])
                    _f1 = float(m[5].item()) if hasattr(m[5], 'item') else float(m[5])
                    _mcc = float(m[6]) if isinstance(m[6], float) else float(m[6])
                    sample_details.append({
                        'name': names[i] if i < len(names) else f'sample_{bi}_{i}',
                        'length': seq_len,
                        'sequence': seq_str[:seq_len],
                        'gt_num_pairs': len(gt_pairs),
                        'pred_num_pairs': len(pred_pairs),
                        'tp': len(tp_pairs),
                        'fp': len(fp_pairs),
                        'fn': len(fn_pairs),
                        'precision': _prec if not np.isnan(_prec) else 0.0,
                        'recall': _rec if not np.isnan(_rec) else 0.0,
                        'f1': _f1 if not np.isnan(_f1) else 0.0,
                        'mcc': _mcc if not np.isnan(_mcc) else 0.0,
                        'dot_bracket_gt': gt_db,
                        'dot_bracket_pred': pred_db,
                        'gt_pairs': gt_pairs[:50],  # 最多保存50对，避免太大
                        'pred_pairs': pred_pairs[:50],
                    })
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'{log_prefix} OOM b{bi} L={set_max_len_val}')
                torch.cuda.empty_cache()
                continue
            raise

    if not all_metrics:
        return None

    arr = np.array(all_metrics)
    acc, prec, rec, sens, spec, f1, mcc = [np.nan_to_num(arr[:, i]).mean() for i in range(7)]
    dt = time.time() - t0
    print(f'{log_prefix} N={len(all_metrics)} F1={f1:.4f} P={prec:.4f} '
          f'R={rec:.4f} MCC={mcc:.4f} ({dt:.1f}s)')

    result = {
        'N': len(all_metrics),
        'F1': float(f1), 'Precision': float(prec), 'Recall': float(rec),
        'MCC': float(mcc), 'Accuracy': float(acc), 'Specificity': float(spec),
    }
    if detailed:
        # 按 F1 排序
        sample_details.sort(key=lambda x: x['f1'], reverse=True)
        result['samples'] = sample_details
    return result


def print_detailed_report(results: Dict[str, Any]):
    """打印逐样本详细报告"""
    for ds_name, res in results.items():
        if 'samples' not in res:
            continue
        samples = res['samples']
        print(f'\n{"="*80}')
        print(f' 数据集: {ds_name}  |  N={res["N"]}  |  平均 F1={res["F1"]:.4f}')
        print(f'{"="*80}')

        # 表头
        print(f'\n{"Rank":<5} {"Name":<30} {"Len":>5} {"F1":>7} {"Prec":>7} '
              f'{"Rec":>7} {"MCC":>7} {"GT_P":>5} {"Pred_P":>6} {"TP":>4} {"FP":>4} {"FN":>4}')
        print('-' * 120)

        for rank, s in enumerate(samples, 1):
            print(f'{rank:<5} {s["name"]:<30} {s["length"]:>5} {s["f1"]:>7.4f} '
                  f'{s["precision"]:>7.4f} {s["recall"]:>7.4f} {s["mcc"]:>7.4f} '
                  f'{s["gt_num_pairs"]:>5} {s["pred_num_pairs"]:>6} '
                  f'{s["tp"]:>4} {s["fp"]:>4} {s["fn"]:>4}')

        # Top 3 最好
        print(f'\n--- Top 3 最好样本 ---')
        for s in samples[:3]:
            print(f'\n  [{s["name"]}] L={s["length"]} F1={s["f1"]:.4f}')
            print(f'  序列: {_format_sequence_display(s["sequence"], s["length"])}')
            print(f'  GT结构:   {_format_sequence_display(s["dot_bracket_gt"], s["length"])}')
            print(f'  Pred结构: {_format_sequence_display(s["dot_bracket_pred"], s["length"])}')
            print(f'  GT配对数={s["gt_num_pairs"]}, Pred配对数={s["pred_num_pairs"]}, '
                  f'TP={s["tp"]}, FP={s["fp"]}, FN={s["fn"]}')

        # Top 3 最差
        print(f'\n--- Top 3 最差样本 ---')
        for s in samples[-3:]:
            print(f'\n  [{s["name"]}] L={s["length"]} F1={s["f1"]:.4f}')
            print(f'  序列: {_format_sequence_display(s["sequence"], s["length"])}')
            print(f'  GT结构:   {_format_sequence_display(s["dot_bracket_gt"], s["length"])}')
            print(f'  Pred结构: {_format_sequence_display(s["dot_bracket_pred"], s["length"])}')
            print(f'  GT配对数={s["gt_num_pairs"]}, Pred配对数={s["pred_num_pairs"]}, '
                  f'TP={s["tp"]}, FP={s["fp"]}, FN={s["fn"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='path to .pt (best/last)')
    parser.add_argument('--test_sets', default='bpRNA,ArchiveII,PDB_TS1,PDB_TS2,PDB_TS3,PDB_TS_hard',
                         help='comma-separated test set names')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=1,
                         help='推理时 seed 投票次数; 1 = 单次')
    parser.add_argument('--physics_beta', type=float, default=0.0,
                         help='物理 guidance 强度 (0 = off)')
    parser.add_argument('--physics_lambda_pk', type=float, default=0.0,
                         help='pseudoknot penalty (0 = 允许 pk)')
    parser.add_argument('--out_json', default=None, help='保存结果到 JSON')
    parser.add_argument('--detailed', action='store_true',
                         help='输出逐样本详细结果 (含序列、GT/Pred配对、dot-bracket)')
    parser.add_argument('--top_k', type=int, default=10,
                         help='详细模式下, JSON 中保存 top-k 最好和最差的样本全部信息')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'[Eval] loading ckpt {args.ckpt}')
    model, cfg = load_model_from_ckpt(args.ckpt, device)
    print(f'[Eval] num_steps={args.num_steps} num_samples={args.num_samples} '
          f'beta={args.physics_beta} lambda_pk={args.physics_lambda_pk}')
    if args.detailed:
        print(f'[Eval] detailed mode ON, top_k={args.top_k}')

    test_names = [s.strip() for s in args.test_sets.split(',') if s.strip()]
    results = {}
    for name in test_names:
        if name not in TEST_SET_FILES:
            print(f'[WARN] unknown test set: {name}')
            continue
        files = TEST_SET_FILES[name]
        print(f'[Eval] ===== {name} =====')
        res = eval_on_dataset(
            model, files, device,
            num_steps=args.num_steps,
            num_samples_per_input=args.num_samples,
            physics_beta=args.physics_beta,
            physics_lambda_pk=args.physics_lambda_pk,
            log_prefix=f'  [{name}]',
            detailed=args.detailed,
        )
        if res is not None:
            results[name] = res

    print('=' * 70)
    print('Summary:')
    print(f'{"Dataset":<20s} {"N":>6s} {"F1":>8s} {"Prec":>8s} {"Rec":>8s} {"MCC":>8s}')
    for name, r in results.items():
        print(f'{name:<20s} {r["N"]:>6d} {r["F1"]:>8.4f} '
              f'{r["Precision"]:>8.4f} {r["Recall"]:>8.4f} {r["MCC"]:>8.4f}')

    # 打印详细报告
    if args.detailed:
        print_detailed_report(results)

    if args.out_json:
        # 如果是 detailed 模式, 对 samples 列表做截断 (保留 top_k 最好 + top_k 最差)
        save_results = {}
        for name, r in results.items():
            r_copy = dict(r)
            if 'samples' in r_copy:
                samples = r_copy['samples']
                top_best = samples[:args.top_k]
                top_worst = samples[-args.top_k:] if len(samples) > args.top_k else []
                r_copy['top_best'] = top_best
                r_copy['top_worst'] = top_worst
                # 只保留 summary 列表 (不含序列, 节省空间)
                r_copy['all_samples_summary'] = [
                    {'name': s['name'], 'length': s['length'], 'f1': s['f1'],
                     'precision': s['precision'], 'recall': s['recall'], 'mcc': s['mcc'],
                     'gt_num_pairs': s['gt_num_pairs'], 'pred_num_pairs': s['pred_num_pairs']}
                    for s in samples
                ]
                del r_copy['samples']
            save_results[name] = r_copy

        out_data = {
            'ckpt': args.ckpt,
            'num_steps': args.num_steps,
            'num_samples': args.num_samples,
            'physics_beta': args.physics_beta,
            'physics_lambda_pk': args.physics_lambda_pk,
            'detailed': args.detailed,
            'results': save_results,
        }
        os.makedirs(os.path.dirname(args.out_json) if os.path.dirname(args.out_json) else '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f'\nresults saved to {args.out_json}')


if __name__ == '__main__':
    main()
