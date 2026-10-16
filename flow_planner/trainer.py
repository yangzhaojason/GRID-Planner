import logging
import os
import time
from typing import Dict, Optional
import argparse
import random
from omegaconf import DictConfig, OmegaConf
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

@hydra.main(version_base=None, config_path="script")
def trainer(cfg: DictConfig):
    
    set_seed(cfg.seed)
    
    global_rank, local_rank, world_size = ddp_setup_universal(verbose=True, cfg=cfg)
    device = torch.device(f"cuda:{local_rank}" if cfg.ddp.distributed else cfg.device)
    cfg.device = str(device)

    if local_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    log_path = os.path.join(cfg.save_dir, f'{cfg.job_name}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    
    assert cfg.train.batch_size >= world_size, f"batch size is at least world size, but got batch size {cfg.train.batch_size} running on {world_size} devices."
    
    if global_rank > 0:
        logger.setLevel(logging.WARNING)
        
    # build model
    logger.info("build model")
    model = instantiate(cfg.model)

    # load pretrain checkpoint
    if cfg.pretrained_checkpoint is not None:
        load_model_weights(model, load_model(cfg.pretrained_checkpoint))

    model = model.to(device)
    if cfg.ddp.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # construct dataset & dataloader
    logger.info("construct dataset and dataloader")
    trainset = instantiate(cfg.data.dataset.train)
    trainsampler = DistributedSampler(trainset, num_replicas=world_size, rank=global_rank, shuffle=True)
    trainloader = DataLoader(trainset, sampler=trainsampler, batch_size=cfg.train.batch_size // world_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_mem, drop_last=True, collate_fn=collect_batch)
    
    # build core first (may contain auxiliary trainable modules like value baseline)
    logger.info("build core")
    core = instantiate(cfg.core)

    # wrap core.value_net with DDP (if present) so critic is synchronized across ranks
    if cfg.ddp.distributed and hasattr(core, "value_net") and core.value_net is not None:
        core.value_net = DDP(core.value_net.to(device), device_ids=[local_rank], output_device=local_rank)

    # build optimizer (model params + core aux params)
    logger.info("build optimizer")
    params = list(ddp.get_model(model).get_optimizer_params())
    if hasattr(core, "get_optimizer_params"):
        params.extend(core.get_optimizer_params())
    optimizer = instantiate(cfg.optimizer, params=params, lr=cfg.optimizer.lr)

    # build scheduler
    logger.info("build scheduler")
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    
    # build ema
    logger.info("build ema")
    ema = instantiate(cfg.ema, model=model)
    
    if cfg.resume_path is not None:
        model, optimizer, scheduler, init_epoch, wandb_id, ema = resume_model(
            cfg.resume_path,
            model,
            optimizer,
            scheduler,
            ema,
            cfg.device,
            core=core,
        )
    else:
        wandb_id = None
        init_epoch = 0
        
    # initialize recorder eg. wandb, tensorboard
    logger.info("initialize recorder")
    recorder_dict = {}
    if hasattr(cfg, "recorder"):
        for name, rec in cfg.recorder.items():
            if name == 'wandb':    
                recorder_dict[name] = instantiate(rec, wandb_id=wandb_id)
            else:
                recorder_dict[name] = instantiate(rec)

    # train model
    logger.info("Training launched")
    
    timer = time.time()

    with tqdm(total=cfg.train.epoch, initial=init_epoch, disable=(global_rank != 0)) as epoch_bar:
        for epoch in range(init_epoch, cfg.train.epoch):
            trainsampler.set_epoch(epoch)
            if hasattr(core, "set_epoch"):
                core.set_epoch(epoch)
                        
            model.train()
            
            loss_list = []
            
            # Training Step
            with tqdm(total=len(trainloader), desc=f"Epoch {epoch+1} - Training", disable=(global_rank != 0), leave=False) as batch_bar:
                for k, data in enumerate(trainloader):
                    data = data.to(cfg.device)
                    loss = core.train_step(model, data)
                    
                    optimizer.zero_grad()
                    loss['total_loss'].backward()

                    clip_params = list(ddp.get_model(model).parameters())
                    if hasattr(core, "value_net") and core.value_net is not None:
                        clip_params.extend(list(ddp.get_model(core.value_net).parameters()))
                    nn.utils.clip_grad_norm_(clip_params, 5)
                    
                    optimizer.step()

                    ema.update(model)

                    loss_list.append(loss)

                    # richer tqdm display (keep short keys to avoid clutter)
                    tqdm_dict = {'tot': f"{loss['total_loss'].item():.3f}"}
                    if 'ego_planning_loss' in loss:
                        tqdm_dict['plan'] = f"{loss['ego_planning_loss'].item():.3f}"
                    if 'consistency_loss' in loss:
                        tqdm_dict['cons'] = f"{loss['consistency_loss'].item():.6f}"
                    if 'gaiw_mean' in loss:
                        tqdm_dict['w'] = f"{loss['gaiw_mean'].item():.2f}"
                    if 'rrs_penalty_mean' in loss:
                        tqdm_dict['rrsP'] = f"{loss['rrs_penalty_mean'].item():.3f}"
                    if 'rrs_safety_pen' in loss:
                        tqdm_dict['safe'] = f"{loss['rrs_safety_pen'].item():.2f}"
                    if 'rrs_rule_pen' in loss:
                        tqdm_dict['rule'] = f"{loss['rrs_rule_pen'].item():.2f}"
                    if 'rrs_comfort_pen' in loss:
                        tqdm_dict['comf'] = f"{loss['rrs_comfort_pen'].item():.2f}"
                    if 'rrs_progress' in loss:
                        tqdm_dict['prog'] = f"{loss['rrs_progress'].item():.2f}"
                    if 'rrs_mean_speed' in loss:
                        tqdm_dict['spd'] = f"{loss['rrs_mean_speed'].item():.2f}"
                    if 'rrs_nu' in loss:
                        tqdm_dict['nu'] = f"{loss['rrs_nu'].item():.3g}"
                    if 'rl_actor_loss' in loss:
                        tqdm_dict['rlA'] = f"{loss['rl_actor_loss'].item():.3f}"
                    if 'rl_v_loss' in loss:
                        tqdm_dict['rlV'] = f"{loss['rl_v_loss'].item():.3f}"
                    if 'rl_adv_mean' in loss:
                        tqdm_dict['adv'] = f"{loss['rl_adv_mean'].item():.3f}"
                    if 'rl_pos_adv_frac' in loss:
                        tqdm_dict['pos'] = f"{loss['rl_pos_adv_frac'].item():.2f}"
                    if 'rl_improvement_mean' in loss:
                        tqdm_dict['imp'] = f"{loss['rl_improvement_mean'].item():.3f}"
                    if 'rl_actor_active_frac' in loss:
                        tqdm_dict['act'] = f"{loss['rl_actor_active_frac'].item():.2f}"
                    if 'rl_auto_scale' in loss:
                        tqdm_dict['shr'] = f"{loss['rl_auto_scale'].item():.2f}"
                    if 'rl_group_best_reward_mean' in loss:
                        tqdm_dict['rbest'] = f"{loss['rl_group_best_reward_mean'].item():.3f}"
                    if 'rl_e_diff_mean' in loss:
                        tqdm_dict['ediff'] = f"{loss['rl_e_diff_mean'].item():.3f}"
                    if 'rl_weight_eff' in loss:
                        tqdm_dict['rw'] = f"{loss['rl_weight_eff'].item():.2f}"
                        
                    batch_bar.set_postfix(tqdm_dict)
                    batch_bar.update(1)

            scheduler.step()
            
            # record loss
            epoch_loss = {name: ( sum([l[name] for l in loss_list]) / len(loss_list) ) for name in loss.keys()}
            epoch_lr = {f"lr/group_{i}": param_group['lr'] for i, param_group in enumerate(optimizer.param_groups)}
            
            if cfg.ddp.distributed:
                epoch_loss = reduce_and_average_losses(epoch_loss, torch.device(cfg.device))

            def _fmt_epoch(name: str) -> str:
                v = epoch_loss.get(name, None)
                if v is None:
                    return ""
                try:
                    return f"{name}={float(v):.4g}"
                except Exception:
                    return ""

            epoch_keys = [
                "total_loss",
                "ego_planning_loss",
                "gaiw_mean",
                "gaiw_p90",
                "gaiw_p99",
                "rrs_penalty_mean",
                "rrs_reward_mean",
                "rl_actor_loss",
                "rl_v_loss",
                "rl_adv_mean",
                "rl_pos_adv_frac",
                "rl_improvement_mean",
                "rl_actor_active_frac",
                "rl_auto_scale",
                "rl_reward_mean",
                "rl_ref_reward_mean",
                "rl_weight_eff",
                "rl_critic_weight_eff",
                "rl_group_best_reward_mean",
                "rl_e_diff_mean",
            ]
            rich = " ".join([s for s in (_fmt_epoch(k) for k in epoch_keys) if s])
            logger.info(f"epoch loss : total={epoch_loss['total_loss']:.3e} | epoch_lr : {epoch_lr['lr/group_0']:.3e} | {rich}")

            if global_rank == 0:
                for recorder in recorder_dict.values():
                    recorder.record_loss(epoch_loss, epoch+1)
                    recorder.record_loss(epoch_lr, epoch+1)
                    
            # save model
            if global_rank == 0 and (epoch+1) % cfg.train.save_utd == 0:
                if 'wandb' in recorder_dict.keys():
                    wandb_id = recorder_dict['wandb'].id
                else:
                    wandb_id = None
                save_model(model, optimizer, scheduler, cfg.save_dir, epoch, epoch_loss['total_loss'], wandb_id, ema.ema, save_every_epoch=cfg.save_every_since, core=core)
                print(f"Model saved in {cfg.save_dir}\n")
            
            epoch_bar.update(1)
            
            if cfg.ddp.distributed:
                torch.cuda.synchronize()
            
    logger.info(f"Training finished - Time consumed: {time.strftime('%H:%M:%S', time.gmtime(time.time()-timer))}")
    
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    
if __name__ == '__main__':
    trainer()
