import logging
import os
import time
import random
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_planner.train_utils import ddp
from flow_planner.data.utils.collect import collect_batch
from flow_planner.train_utils.ddp import reduce_and_average_losses, ddp_setup_universal
from flow_planner.train_utils.save_model import save_model, resume_model, load_model, load_model_weights


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="../script", config_name="flow_planner_standard")
def distill_trainer(cfg: DictConfig):
    """
    Distillation trainer entrypoint for OPSD:
      - builds student model as usual
      - builds teacher model from same config
      - loads teacher weights from cfg.opsd.teacher_ckpt_path
      - trains using OPSDDistillCore (cfg.core should point to that core)
    """
    set_seed(cfg.seed)

    global_rank, local_rank, world_size = ddp_setup_universal(verbose=True, cfg=cfg)
    device = torch.device(f"cuda:{local_rank}" if cfg.ddp.distributed else cfg.device)
    cfg.device = str(device)

    if local_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    log_path = os.path.join(cfg.save_dir, f"{cfg.job_name}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    assert cfg.train.batch_size >= world_size

    if global_rank > 0:
        logger.setLevel(logging.WARNING)

    # build student model
    logger.info("build student model")
    model = instantiate(cfg.model)

    # load pretrain checkpoint (student init)
    if cfg.pretrained_checkpoint is not None:
        load_model_weights(model, load_model(cfg.pretrained_checkpoint))

    model = model.to(device)
    if cfg.ddp.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # dataset / loader
    logger.info("construct dataset and dataloader")
    trainset = instantiate(cfg.data.dataset.train)
    trainsampler = DistributedSampler(trainset, num_replicas=world_size, rank=global_rank, shuffle=True)
    trainloader = DataLoader(
        trainset,
        sampler=trainsampler,
        batch_size=cfg.train.batch_size // world_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
        collate_fn=collect_batch,
    )

    # build core (must be OPSDDistillCore)
    logger.info("build distill core")
    core = instantiate(cfg.core)
    if cfg.ddp.distributed and hasattr(core, "value_net") and core.value_net is not None:
        core.value_net = DDP(core.value_net.to(device), device_ids=[local_rank], output_device=local_rank)

    # optimizer / scheduler
    logger.info("build optimizer and scheduler")
    params = list(ddp.get_model(model).get_optimizer_params())
    if hasattr(core, "get_optimizer_params"):
        params.extend(core.get_optimizer_params())
    optimizer = instantiate(cfg.optimizer, params=params, lr=cfg.optimizer.lr)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # ema (still useful for student checkpointing)
    ema = instantiate(cfg.ema, model=model)

    # build teacher model + attach
    teacher_ckpt = None
    try:
        teacher_ckpt = cfg.core.opsd.teacher_ckpt_path
    except Exception:
        teacher_ckpt = None
    if not teacher_ckpt:
        raise ValueError("Missing core.opsd.teacher_ckpt_path for distillation run. Use core=opds_distill and set core.opsd.teacher_ckpt_path=... (or TEACHER_CKPT env).")
    teacher = instantiate(cfg.model)
    core.attach_teacher(teacher, teacher_ckpt, cfg.device)

    if cfg.resume_path is not None:
        model, optimizer, scheduler, init_epoch, wandb_id, ema = resume_model(
            cfg.resume_path, model, optimizer, scheduler, ema, cfg.device, core=core
        )
    else:
        init_epoch = 0
        wandb_id = None

    logger.info("Distillation training launched")
    timer = time.time()

    with tqdm(total=cfg.train.epoch, initial=init_epoch, disable=(global_rank != 0)) as epoch_bar:
        for epoch in range(init_epoch, cfg.train.epoch):
            trainsampler.set_epoch(epoch)
            if hasattr(core, "set_epoch"):
                core.set_epoch(epoch)

            model.train()
            loss_list = []

            with tqdm(total=len(trainloader), desc=f"Epoch {epoch+1} - Distill", disable=(global_rank != 0), leave=False) as batch_bar:
                for _, data in enumerate(trainloader):
                    data = data.to(cfg.device)
                    loss = core.train_step(model, data)

                    optimizer.zero_grad()
                    loss["total_loss"].backward()
                    clip_params = list(ddp.get_model(model).parameters())
                    if hasattr(core, "value_net") and core.value_net is not None:
                        clip_params.extend(list(ddp.get_model(core.value_net).parameters()))
                    nn.utils.clip_grad_norm_(clip_params, 5)
                    optimizer.step()
                    ema.update(model)

                    loss_list.append(loss)

                    tqdm_dict = {"tot": f"{loss['total_loss'].item():.3f}"}
                    if "ego_planning_loss" in loss:
                        tqdm_dict["plan"] = f"{loss['ego_planning_loss'].item():.3f}"
                    if "opsd_distill_loss" in loss:
                        tqdm_dict["dist"] = f"{loss['opsd_distill_loss'].item():.3f}"
                    if "opsd_distill_weight_eff" in loss:
                        tqdm_dict["dw"] = f"{loss['opsd_distill_weight_eff'].item():.3f}"
                    if "opsd_improvement_mean" in loss:
                        tqdm_dict["imp"] = f"{loss['opsd_improvement_mean'].item():.3f}"
                    if "opsd_teacher_reward_mean" in loss:
                        tqdm_dict["tR"] = f"{loss['opsd_teacher_reward_mean'].item():.3f}"
                    if "opsd_student_reward_mean" in loss:
                        tqdm_dict["sR"] = f"{loss['opsd_student_reward_mean'].item():.3f}"
                    if "rl_actor_loss" in loss:
                        tqdm_dict["rlA"] = f"{loss['rl_actor_loss'].item():.3f}"
                    if "rl_v_loss" in loss:
                        tqdm_dict["rlV"] = f"{loss['rl_v_loss'].item():.3f}"
                    if "rl_pos_adv_frac" in loss:
                        tqdm_dict["pos"] = f"{loss['rl_pos_adv_frac'].item():.2f}"
                    if "opsd_gate" in loss:
                        tqdm_dict["gate"] = f"{loss['opsd_gate'].item():.2f}"
                    batch_bar.set_postfix(tqdm_dict)
                    batch_bar.update(1)

            scheduler.step()

            epoch_loss = {name: (sum([l[name] for l in loss_list]) / len(loss_list)) for name in loss.keys()}
            epoch_lr = {f"lr/group_{i}": pg["lr"] for i, pg in enumerate(optimizer.param_groups)}
            if cfg.ddp.distributed:
                epoch_loss = reduce_and_average_losses(epoch_loss, torch.device(cfg.device))

            # rich epoch logging for OPSD/RL debugging
            def _fmt(k: str) -> str:
                v = epoch_loss.get(k, None)
                if v is None:
                    return ""
                try:
                    return f"{k}={float(v):.4g}"
                except Exception:
                    return ""

            keys = [
                "total_loss",
                "ego_planning_loss",
                "consistency_loss",
                "rrs_penalty_mean",
                "rrs_reward_mean",
                "opsd_distill_loss",
                "opsd_distill_weight_eff",
                "opsd_gate",
                "opsd_improvement_mean",
                "opsd_teacher_reward_mean",
                "opsd_student_reward_mean",
                "opsd_teacher_idx_mean",
                "rl_actor_loss",
                "rl_v_loss",
                "rl_adv_mean",
                "rl_pos_adv_frac",
                "rl_reward_mean",
            ]
            rich = " ".join([s for s in (_fmt(k) for k in keys) if s])
            logger.info(
                f"epoch loss : total={epoch_loss['total_loss']:.3e} | epoch_lr : {epoch_lr['lr/group_0']:.3e} | {rich}"
            )

            if global_rank == 0 and (epoch + 1) % cfg.train.save_utd == 0:
                save_model(
                    model,
                    optimizer,
                    scheduler,
                    cfg.save_dir,
                    epoch,
                    epoch_loss["total_loss"],
                    wandb_id,
                    ema.ema,
                    save_every_epoch=cfg.save_every_since,
                    core=core,
                )
                print(f"Model saved in {cfg.save_dir}\n")

            epoch_bar.update(1)
            if cfg.ddp.distributed:
                torch.cuda.synchronize()

    logger.info(f"Distillation finished - Time consumed: {time.strftime('%H:%M:%S', time.gmtime(time.time()-timer))}")
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    distill_trainer()
