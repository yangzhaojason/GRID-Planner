"""
Quick diagnostic to verify normalization & model predictions.

Usage:
    python -m flow_planner.evaluation.debug_eval \
        --config_path <.hydra/config.yaml> \
        --ckpt_path <model.pth> \
        --data_dir <womd_npz/validation> \
        --data_list <data_list_validation.json>
"""
import argparse, json, torch, logging, numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

from flow_planner.data.dataset.nuplan import NuPlanDataset, NuPlanDataSample
from flow_planner.data.utils.collect import collect_batch

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_list", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config_path)

    # ---- 1. Print normalization stats from training config ----
    ns = cfg.normalization_stats
    log.info("=" * 60)
    log.info("NORMALIZATION STATS (from training config):")
    log.info(f"  ego.uniform.mean = {list(ns.ego.uniform.mean)}")
    log.info(f"  ego.uniform.std  = {list(ns.ego.uniform.std)}")
    log.info(f"  ego_past.mean[:4] = {list(ns.ego_past.mean[:4])}")
    log.info(f"  ego_past.std[:4]  = {list(ns.ego_past.std[:4])}")
    log.info(f"  neighbor_past.mean[:4] = {list(ns.neighbor_past.mean[:4])}")
    log.info(f"  lanes.mean[:4] = {list(ns.lanes.mean[:4])}")
    log.info("=" * 60)

    # ---- 2. Build model & load checkpoint ----
    model = instantiate(cfg.model)
    model = model.to(device)
    model.device = device

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    if 'ema_state_dict' in ckpt:
        raw_sd = ckpt['ema_state_dict']
    elif 'model' in ckpt:
        raw_sd = ckpt['model']
    else:
        raw_sd = ckpt
    sd = {k.replace("module.", "", 1): v for k, v in raw_sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    log.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # ---- 3. Print StateNormalizer internals ----
    sn = model.data_processor.state_normalizer
    log.info(f"StateNormalizer.mean = {sn.mean.squeeze().tolist()}")
    log.info(f"StateNormalizer.std  = {sn.std.squeeze().tolist()}")

    # ---- 4. Load a small batch ----
    dataset = NuPlanDataset(
        data_dir=args.data_dir,
        data_list=args.data_list,
        past_neighbor_num=cfg.model.neighbor_num,
        predicted_neighbor_num=0,
        future_len=cfg.model.future_len,
        future_downsampling_method='uniform',
    )
    subset = Subset(dataset, list(range(min(8, len(dataset)))))
    loader = DataLoader(subset, batch_size=8, shuffle=False,
                        collate_fn=collect_batch, num_workers=0)
    data = next(iter(loader))
    data = data.to(device)
    B = data.ego_current.shape[0]

    gt_raw = data.ego_future.cpu().numpy()  # (B, 80, 3) [x, y, valid]

    # ---- 5. Run single-mode inference (simpler, faster) ----
    with torch.no_grad():
        pred = model.forward_inference(data, use_cfg=True, cfg_weight=None)
    # pred: (B, 1, 80, 2) denormalized

    pred_np = pred.cpu().numpy()

    # ---- 6. Also get normalized prediction (before postprocess) ----
    # Re-run but intercept before postprocess
    with torch.no_grad():
        cfg_flags = torch.cat([
            torch.ones((B,), device=device),
            torch.zeros((B,), device=device)
        ], dim=0).to(torch.int32)
        model_inputs, _ = model.prepare_model_input(
            cfg_flags, data.copy(device), use_cfg=True, is_training=False)
        encoder_inputs = model.extract_encoder_inputs(model_inputs)
        encoder_outputs = model.encoder(**encoder_inputs)
        decoder_extra = model.extract_decoder_inputs(encoder_outputs, model_inputs)
        x_init = torch.randn(
            (B, model.action_num, model.planner_params['action_len'],
             model.planner_params['state_dim']),
            device=device)
        from flow_planner.model.model_utils.traj_tool import assemble_actions
        sample_norm = model.flow_ode.generate(
            x_init, model.decoder, model._model_type,
            use_cfg=True, cfg_weight=None, **decoder_extra)
        sample_norm = assemble_actions(
            sample_norm, model.planner_params['future_len'],
            model.planner_params['action_len'],
            model.planner_params['action_overlap'],
            model.planner_params['state_dim'],
            model.assemble_method)
    norm_np = sample_norm.cpu().numpy()  # (B, 1, 80, 2) NORMALIZED

    # ---- 7. Print diagnostics for first 3 samples ----
    log.info("\n" + "=" * 60)
    log.info("SAMPLE DIAGNOSTICS:")
    log.info("=" * 60)
    for b in range(min(3, B)):
        gt_xy = gt_raw[b, :, :2]
        gt_valid = gt_raw[b, :, 2]
        n_valid = int(gt_valid.sum())
        last_valid = np.where(gt_valid > 0.5)[0][-1] if n_valid > 0 else 79

        pred_xy = pred_np[b, 0, :, :]  # (80, 2) denormalized
        norm_xy = norm_np[b, 0, :, :]  # (80, 2) normalized

        log.info(f"\n--- Sample {b} (valid={n_valid}/80, last_valid_t={last_valid}) ---")
        log.info(f"  GT  endpoint (t={last_valid}): x={gt_xy[last_valid, 0]:.2f}, y={gt_xy[last_valid, 1]:.2f}")
        log.info(f"  GT  range: x=[{gt_xy[:n_valid, 0].min():.2f}, {gt_xy[:n_valid, 0].max():.2f}], "
                 f"y=[{gt_xy[:n_valid, 1].min():.2f}, {gt_xy[:n_valid, 1].max():.2f}]")
        log.info(f"  Pred(norm)  endpoint: x={norm_xy[last_valid, 0]:.4f}, y={norm_xy[last_valid, 1]:.4f}")
        log.info(f"  Pred(norm)  range: x=[{norm_xy[:, 0].min():.4f}, {norm_xy[:, 0].max():.4f}], "
                 f"y=[{norm_xy[:, 1].min():.4f}, {norm_xy[:, 1].max():.4f}]")
        log.info(f"  Pred(denorm) endpoint: x={pred_xy[last_valid, 0]:.2f}, y={pred_xy[last_valid, 1]:.2f}")
        log.info(f"  Pred(denorm) range: x=[{pred_xy[:, 0].min():.2f}, {pred_xy[:, 0].max():.2f}], "
                 f"y=[{pred_xy[:, 1].min():.2f}, {pred_xy[:, 1].max():.2f}]")

        # ADE for this sample
        dists = np.linalg.norm(pred_xy[:n_valid] - gt_xy[:n_valid], axis=-1)
        ade = dists.mean()
        fde = np.linalg.norm(pred_xy[last_valid] - gt_xy[last_valid])
        log.info(f"  ADE={ade:.2f}m, FDE={fde:.2f}m")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
