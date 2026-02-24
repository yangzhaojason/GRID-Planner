"""
WOMD Marginal Motion Prediction evaluation metrics.

Implements: minADE, minFDE, Miss Rate, mAP, Soft mAP.
All metrics are computed per-sample with validity masking (only valid
GT timesteps contribute) and support per-type aggregation
(VEHICLE, PEDESTRIAN, CYCLIST).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


MISS_THRESHOLDS = {
    'VEHICLE': 2.0,
    'PEDESTRIAN': 2.0,
    'CYCLIST': 2.0,
}

MAP_TOLERANCES = [
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
]


def compute_ade(pred: np.ndarray, gt: np.ndarray,
                valid: Optional[np.ndarray] = None) -> float:
    """Average displacement error for a single trajectory.
    pred, gt: (T, 2). valid: (T,) bool — only valid steps count.
    """
    dists = np.linalg.norm(pred - gt, axis=-1)  # (T,)
    if valid is not None and valid.sum() > 0:
        return float(dists[valid].mean())
    return float(dists.mean())


def compute_fde(pred: np.ndarray, gt: np.ndarray,
                valid: Optional[np.ndarray] = None) -> float:
    """Final displacement error at last VALID timestep."""
    if valid is not None and valid.sum() > 0:
        last_valid = np.where(valid)[0][-1]
    else:
        last_valid = len(gt) - 1
    return float(np.linalg.norm(pred[last_valid] - gt[last_valid]))


def compute_min_ade(preds: np.ndarray, gt: np.ndarray,
                    valid: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """preds: (K, T, 2), gt: (T, 2). Returns (minADE, best_mode_index)."""
    ades = [compute_ade(preds[k], gt, valid) for k in range(preds.shape[0])]
    best = int(np.argmin(ades))
    return ades[best], best


def compute_min_fde(preds: np.ndarray, gt: np.ndarray,
                    valid: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """preds: (K, T, 2), gt: (T, 2). Returns (minFDE, best_mode_index)."""
    fdes = [compute_fde(preds[k], gt, valid) for k in range(preds.shape[0])]
    best = int(np.argmin(fdes))
    return fdes[best], best


def compute_miss_rate(preds: np.ndarray, gt: np.ndarray,
                      threshold: float = 2.0,
                      valid: Optional[np.ndarray] = None) -> float:
    """Miss = 1 if minFDE > threshold, else 0."""
    min_fde, _ = compute_min_fde(preds, gt, valid)
    return 1.0 if min_fde > threshold else 0.0


def compute_lateral_longitudinal_errors(
    pred_endpoint: np.ndarray,
    gt_endpoint: np.ndarray,
    gt_heading: float,
) -> Tuple[float, float]:
    diff = pred_endpoint - gt_endpoint
    cos_h = np.cos(gt_heading)
    sin_h = np.sin(gt_heading)
    longitudinal = abs(diff[0] * cos_h + diff[1] * sin_h)
    lateral = abs(-diff[0] * sin_h + diff[1] * cos_h)
    return longitudinal, lateral


def _ap_for_tolerance(confidences: np.ndarray,
                      is_tp: np.ndarray) -> float:
    n = len(confidences)
    if n == 0:
        return 0.0
    sorted_idx = np.argsort(-confidences)
    is_tp_sorted = is_tp[sorted_idx]

    tp_cumsum = np.cumsum(is_tp_sorted)
    total_tp = tp_cumsum[-1]
    if total_tp == 0:
        return 0.0

    precision = tp_cumsum / np.arange(1, n + 1)
    recall = tp_cumsum / total_tp

    ap = 0.0
    for r_threshold in np.linspace(0, 1, 11):
        precs_at_recall = precision[recall >= r_threshold]
        if len(precs_at_recall) > 0:
            ap += np.max(precs_at_recall)
    return ap / 11.0


def _last_valid_idx(valid: Optional[np.ndarray], T: int) -> int:
    if valid is not None and valid.sum() > 0:
        return int(np.where(valid)[0][-1])
    return T - 1


def compute_map(
    all_preds: List[np.ndarray],
    all_gts: List[np.ndarray],
    all_confs: List[np.ndarray],
    all_gt_headings: Optional[List[float]] = None,
    all_valid: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    aps = []
    for lon_tol, lat_tol in MAP_TOLERANCES:
        confidences = []
        is_tp = []

        for i in range(len(all_preds)):
            preds = all_preds[i]
            gt = all_gts[i]
            confs = all_confs[i]
            heading = all_gt_headings[i] if all_gt_headings else 0.0
            valid = all_valid[i] if all_valid else None
            last_t = _last_valid_idx(valid, gt.shape[0])

            for k in range(preds.shape[0]):
                lon_err, lat_err = compute_lateral_longitudinal_errors(
                    preds[k, last_t], gt[last_t], heading)
                tp = (lon_err <= lon_tol) and (lat_err <= lat_tol)
                confidences.append(confs[k])
                is_tp.append(float(tp))

        confidences = np.array(confidences)
        is_tp = np.array(is_tp)
        ap = _ap_for_tolerance(confidences, is_tp)
        aps.append(ap)

    result = {'mAP': float(np.mean(aps))}
    weights = [1.0 / (lon * lat) for lon, lat in MAP_TOLERANCES]
    total_w = sum(weights)
    soft_map = sum(ap * w for ap, w in zip(aps, weights)) / total_w
    result['soft_mAP'] = float(soft_map)
    return result


class WOMDMetricsAccumulator:
    """Accumulates per-sample metrics with validity masking."""

    def __init__(self):
        self._records = defaultdict(list)
        self._preds_for_map = defaultdict(list)
        self._gts_for_map = defaultdict(list)
        self._confs_for_map = defaultdict(list)
        self._headings_for_map = defaultdict(list)
        self._valid_for_map = defaultdict(list)

    def add_sample(self,
                   preds: np.ndarray,
                   gt: np.ndarray,
                   confs: np.ndarray,
                   agent_type: str = 'VEHICLE',
                   gt_heading: float = 0.0,
                   valid: Optional[np.ndarray] = None):
        """
        Args:
            preds: (K, T, 2) predicted trajectories.
            gt: (T, 2) ground truth trajectory.
            confs: (K,) confidence scores.
            valid: (T,) bool — True for valid GT timesteps.
        """
        min_ade, _ = compute_min_ade(preds, gt, valid)
        min_fde, _ = compute_min_fde(preds, gt, valid)
        threshold = MISS_THRESHOLDS.get(agent_type, 2.0)
        miss = compute_miss_rate(preds, gt, threshold, valid)

        for key in ['ALL', agent_type]:
            self._records[key].append({
                'minADE': min_ade,
                'minFDE': min_fde,
                'miss': miss,
            })
            self._preds_for_map[key].append(preds)
            self._gts_for_map[key].append(gt)
            self._confs_for_map[key].append(confs)
            self._headings_for_map[key].append(gt_heading)
            self._valid_for_map[key].append(valid)

    def compute(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for key in self._records:
            records = self._records[key]
            n = len(records)
            if n == 0:
                continue

            res = {
                'minADE': float(np.mean([r['minADE'] for r in records])),
                'minFDE': float(np.mean([r['minFDE'] for r in records])),
                'MissRate': float(np.mean([r['miss'] for r in records])),
                'count': n,
            }

            map_result = compute_map(
                self._preds_for_map[key],
                self._gts_for_map[key],
                self._confs_for_map[key],
                self._headings_for_map[key],
                self._valid_for_map[key],
            )
            res.update(map_result)
            results[key] = res

        return results

    def summary_str(self) -> str:
        results = self.compute()
        lines = []
        for key in ['ALL', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
            if key not in results:
                continue
            r = results[key]
            lines.append(
                f"[{key:>10s}] n={r['count']:>6d}  "
                f"minADE={r['minADE']:.4f}  minFDE={r['minFDE']:.4f}  "
                f"MR={r['MissRate']:.4f}  mAP={r['mAP']:.4f}  "
                f"soft_mAP={r['soft_mAP']:.4f}")
        return '\n'.join(lines)
