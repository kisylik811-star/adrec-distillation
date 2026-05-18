"""
Entry point for consistency distillation of ADRec.

Four modes:
  default            single run: one (use_rccd, use_aw, beta, tau, seed).
  --sweep            (beta, tau) grid sweep with one seed. For hyperparameter
                     selection on ML-100K. Optionally also runs a CD baseline
                     (--include_baseline -> beta=0 = no RCCD).
  --multiseed        SAME configuration with multiple seeds. Writes a
                     consolidated multi-seed JSON for downstream analysis.
  --ablation         4-way ablation on ONE dataset (intended for ML-100K):
                     {CD, CD+RCCD, CD+AW, CD+RCCD+AW} x N seeds. Writes a
                     consolidated ablation JSON.

Teacher checkpoint
------------------
The teacher must be provided via --teacher_ckpt (a .pth state_dict produced
by adrec_original/src/trainer.py). The teacher is constructed with
pretrained=False to avoid the pretrain.pth lookup; the full teacher state
is loaded afterwards.

Always saves
------------
  artifacts_adrec/<dataset>/teacher/teacher.pt          (copy of teacher ckpt)
  artifacts_adrec/<dataset>/teacher/reference.json      (teacher metrics, T)
  artifacts_adrec/<dataset>/teacher/config.json         (teacher arch args)
  artifacts_adrec/<dataset>/<run_name>/{...}            (per-run artifacts)
For --multiseed / --ablation, also writes a consolidated JSON.
"""
import argparse
import json
import logging
import os
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# From the adrec_original tree (caller arranges PYTHONPATH).
from model import Att_Diffuse_model
from utils import Data_Train, Data_Val, Data_Test

from consistency_adrec import ConsistencyADRecStudent
from distill_trainer import (
    distill_train,
    evaluate_at_nfe,
    evaluate_teacher_full_nfe,
    measure_inference_latency,
)
from evaluation import evaluate_teacher_truncated, measure_latency_grid


# ===================================================================
# Dataset metadata (matches trainer.py:item_num_create)
# ===================================================================

ITEM_NUM_BY_DATASET = {
    'ml-100k': 1008,
    'yelp':   64669,
    'sports': 12301,
    'baby':    4731,
    'toys':    7309,
    'beauty':  6086,
}


def parse_args():
    p = argparse.ArgumentParser()

    # ----- Core args (mirror adrec_original/config.yaml defaults) -----
    p.add_argument('--dataset', default='ml-100k',
                   choices=list(ITEM_NUM_BY_DATASET.keys()))
    p.add_argument('--data_root', default='../adrec_original/datasets/data')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--random_seed', type=int, default=2025)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--emb_dropout', type=float, default=0.3)
    p.add_argument('--hidden_act', default='gelu')
    p.add_argument('--dif_blocks', type=int, default=2)
    p.add_argument('--diffusion_steps', type=int, default=32)
    p.add_argument('--noise_schedule', default='trunc_lin')
    p.add_argument('--beta_a', type=float, default=0.3)
    p.add_argument('--beta_b', type=float, default=10.0)
    p.add_argument('--rescale_timesteps', type=lambda s: s.lower() == 'true',
                   default=True)
    p.add_argument('--schedule_sampler_name', default='uniform')
    p.add_argument('--lambda_uncertainty', type=float, default=0.001)
    p.add_argument('--dif_decoder', default='att', choices=['att', 'mlp'])
    p.add_argument('--is_causal', type=lambda s: s.lower() == 'true',
                   default=True)
    p.add_argument('--independent', type=lambda s: s.lower() == 'true',
                   default=True)
    p.add_argument('--cfg_scale', type=float, default=1.0)
    p.add_argument('--geodesic', type=lambda s: s.lower() == 'true',
                   default=False)
    p.add_argument('--parallel_ag', type=lambda s: s.lower() == 'true',
                   default=True)
    p.add_argument('--split_onebyone', type=lambda s: s.lower() == 'true',
                   default=False)
    p.add_argument('--pretrained', type=lambda s: s.lower() == 'true',
                   default=False,
                   help='Forced to False — we load the full teacher ckpt.')
    p.add_argument('--freeze_emb', type=lambda s: s.lower() == 'true',
                   default=False)
    p.add_argument('--loss', default='mse', choices=['mse', 'ce'])
    p.add_argument('--loss_scale', type=float, default=1.0)
    p.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20])
    p.add_argument('--model', default='adrec')  # fixed for compat with Att_Diffuse_model

    # ----- Teacher checkpoint -----
    p.add_argument('--teacher_ckpt', type=str, required=True,
                   help='Path to trained teacher state_dict (.pth/.pt).')

    # ----- Distillation hyperparameters -----
    p.add_argument('--distill_lr', type=float, default=1e-3)
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.999)

    # ----- RCCD -----
    p.add_argument('--contrast_weight', type=float, default=1.0,
                   help='Beta on the RCCD InfoNCE term.')
    p.add_argument('--contrast_temperature', type=float, default=0.1,
                   help='Tau for InfoNCE softmax.')
    p.add_argument('--rccd_num_neg', type=int, default=128,
                   help='K shared catalog negatives per batch.')
    p.add_argument('--use_rccd', type=lambda s: s.lower() == 'true',
                   default=True)
    p.add_argument('--use_adaptive_weighting',
                   type=lambda s: s.lower() == 'true', default=True,
                   help='Apply per-token min-SNR (gamma=5) on L_cons.')

    # ----- Mode selection -----
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--sweep_betas', type=float, nargs='+',
                   default=[0.5, 1.0, 2.0])
    p.add_argument('--sweep_taus', type=float, nargs='+',
                   default=[0.05, 0.1])
    p.add_argument('--include_baseline', action='store_true',
                   help='In sweep, also run beta=0 (CD baseline).')

    p.add_argument('--multiseed', action='store_true',
                   help='Run --use_rccd / --use_adaptive_weighting setting '
                        'across multiple seeds, write consolidated JSON.')
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[1997, 42, 2024])
    p.add_argument('--multiseed_baseline', action='store_true',
                   help='In multiseed, also run a CD-only baseline '
                        '(use_rccd=False, use_aw=False) for each seed.')

    p.add_argument('--ablation', action='store_true',
                   help='Run 4 ablation configs across seeds on one dataset.')

    p.add_argument('--nfe_grid', type=int, nargs='+',
                   default=[1, 2, 4, 8],
                   help='NFEs evaluated in multiseed/ablation modes.')
    p.add_argument('--out_multiseed_json', type=str, default=None)
    p.add_argument('--out_ablation_json', type=str, default=None)

    # ----- Logging -----
    p.add_argument('--log_file', default='log_adrec/')
    p.add_argument('--description', default='RCCD_AW')

    args = p.parse_args()
    # Force pretrained=False so Att_Diffuse_model constructor does not try
    # to load saved/pretrain/<dataset>/pretrain.pth. We load the full
    # teacher state_dict below instead.
    args.pretrained = False
    return args


# ===================================================================
# Boilerplate
# ===================================================================

def fix_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    cudnn.deterministic = True
    cudnn.benchmark = False


def setup_logging(args, suffix=''):
    log_dir = os.path.join(args.log_file, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    fname = os.path.join(
        log_dir,
        f'distill_{suffix}{stamp}.log' if suffix else f'distill_{stamp}.log',
    )
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, filename=fname,
                        format='%(asctime)s - %(message)s', filemode='w')
    return logging.getLogger(__name__)


def load_data(args):
    """Load <dataset>/dataset.pkl, build train/val/test loaders.

    args.item_num is set as a side effect (matches the teacher's pipeline).
    """
    path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Cannot find {path}. Pass --data_root pointing at the directory '
            f'containing <dataset>/dataset.pkl.'
        )
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)

    args.item_num = ITEM_NUM_BY_DATASET[args.dataset]

    tra = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val = Data_Val(data_raw['train'], data_raw['val'],
                   args).get_pytorch_dataloaders()
    tst = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                    args).get_pytorch_dataloaders()
    return tra, val, tst, data_raw


def load_teacher(args, val_loader, test_loader, logger):
    """Build Att_Diffuse_model, load checkpoint, evaluate, persist reference.

    Returns (teacher, teacher_metrics_test, teacher_metrics_val, teacher_ms).
    """
    if not os.path.exists(args.teacher_ckpt):
        raise FileNotFoundError(f'Teacher ckpt not found: {args.teacher_ckpt}')

    device = torch.device(args.device)
    # Att_Diffuse_model needs pretrained=False here (set in parse_args).
    teacher = Att_Diffuse_model(args).to(device)

    print(f'Loading teacher from {args.teacher_ckpt}')
    logger.info(f'Loading teacher from {args.teacher_ckpt}')
    state = torch.load(args.teacher_ckpt, map_location=device,
                       weights_only=False)
    # Some checkpoints are wrapped (e.g. {'model_state_dict': ...}); handle.
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing:
        print(f'[Teacher] missing keys: {missing}')
        logger.info(f'[Teacher] missing keys: {missing}')
    if unexpected:
        print(f'[Teacher] unexpected keys: {unexpected}')
        logger.info(f'[Teacher] unexpected keys: {unexpected}')

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Teacher reference: HR/NDCG at full NFE.
    print('\n=== Teacher (NFE=T) Test Performance ===')
    logger.info('=== Teacher (NFE=T) Test Performance ===')
    teacher_metrics_test = evaluate_teacher_full_nfe(teacher, test_loader, device)
    teacher_metrics_val = evaluate_teacher_full_nfe(teacher, val_loader, device)
    print(teacher_metrics_test)
    logger.info(teacher_metrics_test)

    sample_batch = next(iter(test_loader))
    teacher_ms = measure_inference_latency(teacher, sample_batch, device,
                                           mode='teacher')
    print(f'Teacher latency (NFE=T): {teacher_ms:.4f} ms/sample')
    logger.info(f'Teacher latency (NFE=T): {teacher_ms:.4f} ms/sample')

    # Persist a teacher reference to a canonical path.
    teacher_ref_dir = Path('artifacts_adrec') / args.dataset / 'teacher'
    teacher_ref_dir.mkdir(parents=True, exist_ok=True)

    teacher_ckpt_copy = teacher_ref_dir / 'teacher.pt'
    if not teacher_ckpt_copy.exists():
        shutil.copy(args.teacher_ckpt, teacher_ckpt_copy)
        print(f'[Save] teacher checkpoint copy -> {teacher_ckpt_copy}')

    teacher_config = {
        k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
            else str(v))
        for k, v in vars(args).items()
    }
    with open(teacher_ref_dir / 'config.json', 'w') as f:
        json.dump(teacher_config, f, indent=2)

    with open(teacher_ref_dir / 'reference.json', 'w') as f:
        json.dump({
            'metrics_full_nfe_test': teacher_metrics_test,
            'metrics_full_nfe_val':  teacher_metrics_val,
            'latency_ms':            teacher_ms,
            'T':                     args.diffusion_steps,
        }, f, indent=2)

    return teacher, teacher_metrics_test, teacher_metrics_val, teacher_ms


# ===================================================================
# Run helpers
# ===================================================================

def _run_one(args, teacher, tra, val, tst, logger,
             use_rccd, use_aw, beta, tau, seed, run_name=None):
    """Train a single student configuration. Returns (best_student, run_name)."""
    args.use_rccd = use_rccd
    args.use_adaptive_weighting = use_aw
    args.contrast_weight = beta
    args.contrast_temperature = tau
    args.random_seed = seed

    print(f'\n{"="*70}')
    print(f'Config: use_rccd={use_rccd}, use_aw={use_aw}, '
          f'beta={beta}, tau={tau}, seed={seed}')
    print(f'{"="*70}')
    logger.info(f'Config: use_rccd={use_rccd}, use_aw={use_aw}, '
                f'beta={beta}, tau={tau}, seed={seed}')

    fix_seed(seed)
    student = ConsistencyADRecStudent(teacher, args, ema_decay=args.ema_decay)

    if run_name is None:
        rccd_tag = 'rccd' if use_rccd else 'norccd'
        aw_tag = 'aw' if use_aw else 'noaw'
        run_name = (f"seed{seed}_{rccd_tag}_{aw_tag}"
                    f"_beta{beta}_tau{tau}")

    log_dir = os.path.join('logs_adrec', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f'{run_name}.csv')

    best_student = distill_train(
        student, teacher.diffu, tra, val, tst, args, logger,
        log_csv_path=csv_path, run_name=run_name,
    )
    return best_student, run_name


def _eval_at_all_nfe(student, val_loader, test_loader, nfe_grid, device):
    """Return {'_val': {nfe: m}, '<nfe>': m, ...}."""
    out = {'_val': {}}
    for nfe in nfe_grid:
        m_test = evaluate_at_nfe(student, test_loader, num_steps=nfe,
                                 device=device)
        m_val = evaluate_at_nfe(student, val_loader, num_steps=nfe,
                                device=device)
        out[str(nfe)] = m_test
        out['_val'][str(nfe)] = m_val
    return out


def _teacher_truncated_grid(teacher, val_loader, test_loader, nfe_grid, device):
    """Evaluate truncated-DDPM teacher at every NFE on val and test."""
    base_test, base_val = {}, {}
    for nfe in nfe_grid:
        m_test = evaluate_teacher_truncated(teacher, test_loader,
                                            num_steps=nfe, device=device)
        m_val = evaluate_teacher_truncated(teacher, val_loader,
                                           num_steps=nfe, device=device)
        base_test[str(nfe)] = m_test
        base_val[str(nfe)] = m_val
        print(f'  truncated NFE={nfe}: test={m_test}')
    return base_test, base_val


# ===================================================================
# Modes
# ===================================================================

def run_sweep(args, teacher, tra, val, tst, logger):
    """Single-seed (beta, tau) grid. Optional CD baseline (beta=0)."""
    configs = []
    if args.include_baseline:
        # beta=0 with RCCD on still computes contrast_loss but it doesn't
        # enter the total loss because of the use_rccd=False switch path.
        configs.append(('baseline_cd', False, args.use_adaptive_weighting,
                        0.0, 0.1))
    for beta in args.sweep_betas:
        for tau in args.sweep_taus:
            configs.append((f'beta{beta}_tau{tau}', True,
                            args.use_adaptive_weighting, beta, tau))

    print(f'\n=== Sweep mode: {len(configs)} configurations '
          f'(use_aw={args.use_adaptive_weighting}) ===')
    logger.info(f'Sweep mode: {len(configs)} configs')

    for i, (_, use_rccd, use_aw, beta, tau) in enumerate(configs):
        print(f'\n[Sweep {i+1}/{len(configs)}]')
        logger.info(f'Sweep {i+1}/{len(configs)}')
        _run_one(args, teacher, tra, val, tst, logger,
                 use_rccd=use_rccd, use_aw=use_aw,
                 beta=beta, tau=tau, seed=args.random_seed)


def run_multiseed(args, teacher, tra, val, tst,
                  teacher_test, teacher_val, teacher_ms, logger):
    """Fixed config across seeds. Optional CD baseline alongside.

    Writes a consolidated JSON compatible with the consistency_diffurec
    analyze.py schema (`teacher`, `baseline`, `students`,
    `students_baseline`, `latency`).
    """
    device = torch.device(args.device)

    print('\n[Truncated DDPM teacher] evaluating at varying NFE')
    logger.info('[Truncated DDPM teacher] evaluating at varying NFE')
    baseline_test, baseline_val = _teacher_truncated_grid(
        teacher, val, tst, args.nfe_grid, device,
    )

    # Variants per seed.
    seed_configs = [('main', True, args.use_rccd, args.use_adaptive_weighting,
                     args.contrast_weight, args.contrast_temperature)]
    if args.multiseed_baseline:
        # CD-only baseline: no RCCD, no adaptive weighting.
        seed_configs.append(('baseline', False, False, False,
                             0.0, 0.1))

    results = {
        'dataset': args.dataset,
        'config': {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, list,
                                     type(None)))},
        'teacher': {
            'full_nfe':     teacher_test,
            'full_nfe_val': teacher_val,
            'T':            args.diffusion_steps,
        },
        'baseline':     baseline_test,
        'baseline_val': baseline_val,
        'students':          {},
        'students_baseline': {},
        'latency': {},
    }

    final_student_for_latency = None
    for seed in args.seeds:
        print(f'\n========== Seed {seed} ==========')
        logger.info(f'========== Seed {seed} ==========')

        for variant_name, _, use_rccd, use_aw, beta, tau in seed_configs:
            run_name = (f"seed{seed}"
                        f"_{'rccd' if use_rccd else 'norccd'}"
                        f"_{'aw' if use_aw else 'noaw'}"
                        f"_beta{beta}_tau{tau}")
            if variant_name == 'baseline':
                run_name += '_baseline'

            best_student, _ = _run_one(
                args, teacher, tra, val, tst, logger,
                use_rccd=use_rccd, use_aw=use_aw,
                beta=beta, tau=tau, seed=seed,
                run_name=run_name,
            )

            seed_data = {'_run_name': run_name}
            seed_data.update(_eval_at_all_nfe(best_student, val, tst,
                                              args.nfe_grid, device))
            print(f'  [{variant_name}] seed={seed} '
                  f'test_NFE1={seed_data["1"]}')

            store_key = 'students' if variant_name == 'main' \
                                   else 'students_baseline'
            results[store_key][str(seed)] = seed_data
            final_student_for_latency = best_student

    # Latency grid using the last trained student (architecture is identical
    # across seeds, so the choice does not matter).
    if final_student_for_latency is not None:
        print('\n[Latency] measuring grid')
        sample_batch = next(iter(tst))
        results['latency'] = measure_latency_grid(
            teacher, final_student_for_latency, sample_batch, device,
            args.nfe_grid,
        )

    out_path = args.out_multiseed_json or os.path.join(
        'results_adrec', f'multiseed_{args.dataset}.json'
    )
    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] multi-seed results -> {out_path}')
    logger.info(f'[Done] multi-seed results -> {out_path}')


def run_ablation(args, teacher, tra, val, tst,
                 teacher_test, teacher_val, teacher_ms, logger):
    """4-way ablation across seeds on one dataset.

    Variants:
      cd          : use_rccd=False, use_aw=False
      cd_rccd     : use_rccd=True,  use_aw=False
      cd_aw       : use_rccd=False, use_aw=True
      cd_rccd_aw  : use_rccd=True,  use_aw=True   (the full method)

    Beta/tau are taken from args.contrast_weight / args.contrast_temperature
    (the values selected via --sweep on ML-100K).
    """
    device = torch.device(args.device)

    print('\n[Truncated DDPM teacher] evaluating at varying NFE')
    baseline_test, baseline_val = _teacher_truncated_grid(
        teacher, val, tst, args.nfe_grid, device,
    )

    variants = [
        ('cd',         False, False),
        ('cd_rccd',    True,  False),
        ('cd_aw',      False, True),
        ('cd_rccd_aw', True,  True),
    ]

    results = {
        'dataset': args.dataset,
        'config': {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, list,
                                     type(None)))},
        'teacher': {
            'full_nfe':     teacher_test,
            'full_nfe_val': teacher_val,
            'T':            args.diffusion_steps,
        },
        'baseline':     baseline_test,
        'baseline_val': baseline_val,
        'ablation': {v[0]: {} for v in variants},
        'latency': {},
    }

    final_student_for_latency = None
    for seed in args.seeds:
        print(f'\n========== Seed {seed} ==========')
        for variant_name, use_rccd, use_aw in variants:
            run_name = (f"seed{seed}_{variant_name}"
                        f"_beta{args.contrast_weight}"
                        f"_tau{args.contrast_temperature}")
            best_student, _ = _run_one(
                args, teacher, tra, val, tst, logger,
                use_rccd=use_rccd, use_aw=use_aw,
                beta=args.contrast_weight,
                tau=args.contrast_temperature,
                seed=seed,
                run_name=run_name,
            )
            seed_data = {'_run_name': run_name}
            seed_data.update(_eval_at_all_nfe(best_student, val, tst,
                                              args.nfe_grid, device))
            print(f'  [{variant_name}] seed={seed} '
                  f'test_NFE1={seed_data["1"]}')
            results['ablation'][variant_name][str(seed)] = seed_data
            final_student_for_latency = best_student

    if final_student_for_latency is not None:
        sample_batch = next(iter(tst))
        results['latency'] = measure_latency_grid(
            teacher, final_student_for_latency, sample_batch, device,
            args.nfe_grid,
        )

    out_path = args.out_ablation_json or os.path.join(
        'results_adrec', f'ablation_{args.dataset}.json'
    )
    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] ablation results -> {out_path}')
    logger.info(f'[Done] ablation results -> {out_path}')


# ===================================================================
# Main
# ===================================================================

def main():
    args = parse_args()
    print(args)

    mode_flags = [args.sweep, args.multiseed, args.ablation]
    if sum(mode_flags) > 1:
        raise SystemExit('--sweep, --multiseed, --ablation are mutually '
                         'exclusive.')

    if args.sweep:
        suffix = 'sweep_'
    elif args.multiseed:
        suffix = 'multiseed_'
    elif args.ablation:
        suffix = 'ablation_'
    else:
        suffix = ''
    logger = setup_logging(args, suffix=suffix)
    logger.info(args)

    fix_seed(args.random_seed)
    tra, val, tst, _ = load_data(args)
    print(f'[Data] {args.dataset}: item_num={args.item_num}')
    logger.info(f'[Data] {args.dataset}: item_num={args.item_num}')

    teacher, teacher_test, teacher_val, teacher_ms = load_teacher(
        args, val, tst, logger,
    )

    if args.sweep:
        run_sweep(args, teacher, tra, val, tst, logger)
    elif args.multiseed:
        run_multiseed(args, teacher, tra, val, tst,
                      teacher_test, teacher_val, teacher_ms, logger)
    elif args.ablation:
        run_ablation(args, teacher, tra, val, tst,
                     teacher_test, teacher_val, teacher_ms, logger)
    else:
        _run_one(args, teacher, tra, val, tst, logger,
                 use_rccd=args.use_rccd,
                 use_aw=args.use_adaptive_weighting,
                 beta=args.contrast_weight,
                 tau=args.contrast_temperature,
                 seed=args.random_seed)


if __name__ == '__main__':
    main()