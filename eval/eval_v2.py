#!/usr/bin/env python3
"""
SymFold v2 评估脚本

用法:
    cd /root/aigame/dannyyan/RNADiffFold/symfold
    source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260
    nohup python -u scripts/eval_v2.py --ckpt model/260519-132200-v2-fresh/best.pt \
        --output output/260519-132200-v2-fresh/eval_best.json \
        >> logs/260519-132200-v2-fresh-eval.log 2>&1 &
"""
from __future__ import annotations

import sys
import os
import json
import time
import argparse
import logging

import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYMFOLD_ROOT = os.path.dirname(SCRIPT_DIR)
SYMFOLD_SRC = os.path.join(SYMFOLD_ROOT, 'src')
for p in (SYMFOLD_ROOT, SYMFOLD_SRC,
          os.path.join(SYMFOLD_SRC, 'models', 'condition', 'fm_conditioner')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.v2.model import SymFoldModel_v2 as SymFoldModel
from src.data import (SimpleRNADataset, BucketBatchSampler,
                       build_index, simple_collate_fn)
from src.gpu_features import get_data_fcn_gpu
from common.data_utils import contact_map_masks
from common.loss_utils import rna_evaluation


# 测试集映射
TEST_SETS = {
    'bpRNA':        ['data/bpRNA/TS0.cPickle'],
    'RNAStrAlign':  ['data/RNAStrAlign/test.cPickle'],
    'ArchiveII':    ['data/ArchiveII/archiveII.cPickle'],
    'PDB_TS1':      ['data/PDB/TS1.cPickle'],
    'PDB_TS2':      ['data/PDB/TS2.cPickle'],
    'PDB_TS3':      ['data/PDB/TS3.cPickle'],
    'PDB_TS_hard':  ['data/PDB/TS_hard.cPickle'],
}

BS_TABLE = {80: 128, 160: 64, 240: 32, 320: 16, 400: 12, 480: 8, 560: 6, 640: 4}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger('EvalV2')


def parse_args():
    parser = argparse.ArgumentParser(description='SymFold v2 Evaluation')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to best.pt checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: auto from ckpt path)')
    parser.add_argument('--test_sets', type=str, default='all',
                        help='Comma-separated test set names, or "all"')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Sampling steps (default: 20)')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def load_model(ckpt_path, device, logger):
    logger.info(f'Loading checkpoint: {ckpt_path}')
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck['config']
    mc = cfg['model']

    model = SymFoldModel(
        hidden_dim=mc['hidden_dim'], num_heads=mc['num_heads'],
        dim_head=mc['dim_head'],
        num_layers_enc=mc.get('num_layers_enc', 3),
        num_layers_mid=mc.get('num_layers_mid', 2),
        num_layers_dec=mc.get('num_layers_dec', 3),
        patch_size=mc['patch_size'], cond_dim=mc['cond_dim'],
        max_len=mc['max_len'], dp_rate=0.0,
        rho_0=mc['rho_0'], pos_weight_scale=mc['pos_weight_scale'],
        u_ckpt=mc['u_conditioner_ckpt'],
        num_families=mc.get('num_families', 0),
        local_bias_layers=mc.get('local_bias_layers', 2),
        local_window=mc.get('local_window', 8),
        project_mode=mc.get('project_mode', 'relaxed'),
        max_pairs_per_row=mc.get('max_pairs_per_row', 2),
    )
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    logger.info(f'Model loaded. Epoch={ck.get("epoch","?")}, Val F1={ck.get("val_f1","?")}')
    return model, ck


def evaluate_dataset(model, ds_name, files, device, num_steps, logger):
    """评估单个数据集"""
    alphabet = model.get_alphabet()
    roots = [os.path.join(SYMFOLD_ROOT, f) for f in files]
    idx = build_index(roots, verbose=False)
    sampler = BucketBatchSampler(idx, batch_size_table=BS_TABLE, shuffle=False,
                                  max_set_len=640, seed=2026)
    collate = partial(simple_collate_fn, alphabet=alphabet)
    loader = DataLoader(SimpleRNADataset(idx), batch_sampler=sampler,
                        collate_fn=collate, num_workers=0)

    all_metrics = []
    all_details = []
    t0 = time.time()

    logger.info(f'[{ds_name}] Starting eval ({len(sampler)} batches, {len(idx)} samples)')

    with torch.no_grad():
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
                    seq_oh=seq_enc, num_steps=num_steps,
                    num_samples_per_input=1,
                    physics_beta=0.0, physics_lambda_pk=0.0,
                    physics_alpha_stack=1.0)
                pred = pred.cpu().float()
                for i in range(contact.shape[0]):
                    m = rna_evaluation(pred[i].squeeze(), contact.float()[i].squeeze())
                    all_metrics.append(m)
                    rna_name = batch['names'][i] if 'names' in batch else f'sample_{len(all_metrics)}'
                    seq_len = int(length[i].item())
                    acc, prec_s, rec_s, sens_s, spec_s, f1_s, mcc_s = m
                    all_details.append({
                        'name': rna_name, 'length': seq_len,
                        'f1': float(f1_s), 'precision': float(prec_s),
                        'recall': float(rec_s), 'mcc': float(mcc_s),
                    })

                if (bi + 1) % 20 == 0:
                    logger.info(f'[{ds_name}] batch {bi+1}/{len(sampler)}')

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f'[{ds_name}] OOM at batch {bi}, skipping')
                    torch.cuda.empty_cache()
                    continue
                raise

    elapsed = time.time() - t0
    arr = np.array(all_metrics)
    acc, prec, rec, sens, spec, f1, mcc = [float(np.nan_to_num(arr[:, i]).mean()) for i in range(7)]

    # Top-5 / Bottom-5
    sorted_details = sorted(all_details, key=lambda x: x['f1'])
    bottom5 = sorted_details[:5]
    top5 = sorted_details[-5:]

    result = {
        'dataset': ds_name,
        'num_samples': len(all_metrics),
        'mean_f1': f1, 'mean_precision': prec,
        'mean_recall': rec, 'mean_mcc': mcc,
        'time_sec': elapsed,
        'top5': top5, 'bottom5': bottom5,
    }

    logger.info(f'[{ds_name}] DONE: N={len(all_metrics)} F1={f1:.4f} P={prec:.4f} '
                f'R={rec:.4f} MCC={mcc:.4f} ({elapsed:.1f}s)')
    logger.info(f'[{ds_name}] Top-5:')
    for d in top5:
        logger.info(f'  {d["name"]} L={d["length"]} F1={d["f1"]:.4f}')
    logger.info(f'[{ds_name}] Bottom-5:')
    for d in bottom5:
        logger.info(f'  {d["name"]} L={d["length"]} F1={d["f1"]:.4f}')

    return result


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info('=' * 60)
    logger.info(f'SymFold v2 Evaluation')
    logger.info(f'Checkpoint: {args.ckpt}')
    logger.info(f'Test sets: {args.test_sets}')
    logger.info(f'Num steps: {args.num_steps}')
    logger.info('=' * 60)

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, ck = load_model(args.ckpt, device, logger)

    # 确定测试集
    if args.test_sets == 'all':
        test_sets = TEST_SETS
    else:
        names = [s.strip() for s in args.test_sets.split(',')]
        test_sets = {k: v for k, v in TEST_SETS.items() if k in names}

    # 逐个评估
    all_results = {}
    total_t0 = time.time()
    for ds_name, files in test_sets.items():
        result = evaluate_dataset(model, ds_name, files, device, args.num_steps, logger)
        all_results[ds_name] = result

    total_time = time.time() - total_t0

    # 汇总
    logger.info('=' * 60)
    logger.info('SUMMARY')
    logger.info(f'{"Dataset":<15} {"N":>6} {"F1":>8} {"Prec":>8} {"Recall":>8} {"MCC":>8}')
    logger.info('-' * 60)
    for ds_name, r in all_results.items():
        logger.info(f'{ds_name:<15} {r["num_samples"]:>6} {r["mean_f1"]:>8.4f} '
                    f'{r["mean_precision"]:>8.4f} {r["mean_recall"]:>8.4f} {r["mean_mcc"]:>8.4f}')
    logger.info('-' * 60)
    logger.info(f'Total time: {total_time:.1f}s')
    logger.info('=' * 60)

    # 保存
    output_path = args.output
    if output_path is None:
        ckpt_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
        output_path = os.path.join(SYMFOLD_ROOT, 'output', '260519-132200-v2-fresh', 'eval_best.json')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        'checkpoint': args.ckpt,
        'epoch': ck.get('epoch'),
        'val_f1': ck.get('val_f1'),
        'num_steps': args.num_steps,
        'total_time_sec': total_time,
        'results': all_results,
    }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f'Results saved to: {output_path}')


if __name__ == '__main__':
    main()
