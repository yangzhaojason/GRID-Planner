"""
WOMD Motion Prediction Evaluation Script.

Loads a trained GRID-Planner checkpoint, iterates over preprocessed WOMD
validation NPZ files, runs multimodal inference (N=64 oversample → K=6 NMS),
and computes official metrics (minADE, minFDE, Miss Rate, mAP, Soft mAP).

Supports multi-GPU evaluation: each GPU processes a shard of the data in
a separate process, results are merged at the end.

Usage:
    python -m flow_planner.evaluation.womd_eval \
        --config_path /path/to/.hydra/config.yaml \
        --ckpt_path /path/to/checkpoint.pth \
        --data_dir /path/to/womd_npz/validation \
        --data_list /path/to/womd_npz/validation/data_list_validation.json \
        --num_samples 64 \
        --num_modes 6
"""

import os
import json
import argparse
import logging
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from omegaconf import OmegaConf
from hydra.utils import instantiate

from flow_planner.data.dataset.nuplan import NuPlanDataset, NuPlanDataSample
from flow_planner.evaluation.womd_metrics import WOMDMetricsAccumulator

logger = logging.getLogger(__name__)

OBJECT_TYPES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']


def collate_fn(batch):
    """Custom collate: stack NuPlanDataSample fields into batched sample."""
    keys = [
        'ego_past', 'ego_current', 'ego_future',
        'neighbor_past', 'neighbor_future', 'neighbor_future_observed',
        'lanes', 'lanes_speedlimit', 'lanes_has_speedlimit',
        'routes', 'routes_speedlimit', 'routes_has_speedlimit',
        'map_objects',
    ]
    batched = {}
    for k in keys:
        tensors = [getattr(s, k) for s in batch]
        batched[k] = torch.stack(tensors, dim=0)

    return NuPlanDataSample(
        batched=True,
        **{k: batched[k] for k in keys if k != 'neighbor_future'},
        neighbor_future=batched['neighbor_future'],
    )


def _eval_shard(gpu_id, num_gpus, args, return_dict):
    """Run evaluation on a single GPU for a shard of the dataset."""
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [GPU{gpu_id}] %(levelname)s %(message)s")
    log = logging.getLogger(f"eval_gpu{gpu_id}")

    device = torch.device(f'cuda:{gpu_id}')

    cfg = OmegaConf.load(args.config_path)
    model = instantiate(cfg.model)
    model = model.to(device)
    model.device = device

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    # Prefer EMA weights (better for evaluation), fallback to training weights
    if 'ema_state_dict' in ckpt:
        raw_sd = ckpt['ema_state_dict']
    elif 'model' in ckpt:
        raw_sd = ckpt['model']
    else:
        raw_sd = ckpt
    # Strip DDP "module." prefix if present
    sd = {k.replace("module.", "", 1): v for k, v in raw_sd.items()}
    model.load_state_dict(sd, strict=True)
    log.info(f"Loaded {len(sd)} parameters (epoch {ckpt.get('epoch', '?')})")
    model.eval()

    dataset = NuPlanDataset(
        data_dir=args.data_dir,
        data_list=args.data_list,
        past_neighbor_num=cfg.model.neighbor_num,
        predicted_neighbor_num=0,
        future_len=cfg.model.future_len,
        future_downsampling_method='uniform',
    )

    total = len(dataset)
    if args.max_samples and args.max_samples > 0:
        total = min(total, args.max_samples)
    shard_size = (total + num_gpus - 1) // num_gpus
    start_idx = gpu_id * shard_size
    end_idx = min(start_idx + shard_size, total)
    indices = list(range(start_idx, end_idx))

    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True)

    metrics = WOMDMetricsAccumulator()
    data_list = json.load(open(args.data_list))

    log.info(f"Shard [{start_idx}:{end_idx}] = {len(indices)} samples")
    t0 = time.time()
    processed = 0

    with torch.no_grad():
        for batch_i, data in enumerate(dataloader):
            data = data.to(device)
            B = data.ego_current.shape[0]

            trajs, confs = model.forward_inference_multimodal(
                data,
                num_samples=args.num_samples,
                num_modes=args.num_modes,
                use_cfg=True)

            trajs_np = trajs.cpu().numpy()
            confs_np = confs.cpu().numpy()
            gt_np = data.ego_future.cpu().numpy()  # (B, T, 3)

            for b in range(B):
                gt_xy = gt_np[b, :, :2]       # (T, 2)
                pred_xy = trajs_np[b, :, :, :2]  # (K, T, 2)
                conf = confs_np[b]             # (K,)

                # dim 2 stores validity flag (1=valid, 0=invalid)
                gt_valid = gt_np[b, :, 2] > 0.5  # (T,) bool

                metrics.add_sample(
                    preds=pred_xy, gt=gt_xy, confs=conf,
                    agent_type='VEHICLE',
                    valid=gt_valid)

                processed += 1

            if (batch_i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1)
                eta = (len(indices) - processed) / max(rate, 1e-6)
                log.info(f"{processed}/{len(indices)} | "
                         f"{rate:.1f} samples/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log.info(f"Done: {processed} samples in {elapsed:.0f}s")

    return_dict[gpu_id] = metrics


def run_evaluation(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("No CUDA devices available")
        return

    logger.info(f"Running evaluation on {num_gpus} GPU(s)")

    if num_gpus == 1:
        return_dict = {}
        _eval_shard(0, 1, args, return_dict)
        merged = return_dict[0]
    else:
        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=_eval_shard,
                           args=(gpu_id, num_gpus, args, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge metrics from all shards
        merged = WOMDMetricsAccumulator()
        for gpu_id in range(num_gpus):
            if gpu_id in return_dict:
                shard_metrics = return_dict[gpu_id]
                for key in shard_metrics._records:
                    merged._records[key].extend(shard_metrics._records[key])
                    merged._preds_for_map[key].extend(shard_metrics._preds_for_map[key])
                    merged._gts_for_map[key].extend(shard_metrics._gts_for_map[key])
                    merged._confs_for_map[key].extend(shard_metrics._confs_for_map[key])
                    merged._headings_for_map[key].extend(shard_metrics._headings_for_map[key])

    logger.info("=" * 60)
    logger.info("WOMD Motion Prediction Results:")
    logger.info("=" * 60)
    logger.info("\n" + merged.summary_str())

    if args.output_path:
        results = merged.compute()
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="WOMD Motion Prediction Evaluation")
    parser.add_argument("--config_path", required=True,
                        help="Path to .hydra/config.yaml from training.")
    parser.add_argument("--ckpt_path", required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--data_dir", required=True,
                        help="Path to preprocessed WOMD NPZ directory.")
    parser.add_argument("--data_list", required=True,
                        help="Path to data_list JSON.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of oversampled trajectories per agent.")
    parser.add_argument("--num_modes", type=int, default=6,
                        help="Number of modes after NMS.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save JSON results.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Evaluate only first N samples (0 = all).")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
