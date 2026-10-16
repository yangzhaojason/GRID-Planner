import copy
import torch
import os
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from flow_planner.train_utils.ddp import get_model


def load_model_weights(model, checkpoint, use_ema=True):
    """Load plain or DDP weights into either a plain or DDP model.

    Accept training checkpoints and raw state dictionaries, retaining strict
    parameter validation so incompatible architectures fail at the source.
    """
    if use_ema and "ema_state_dict" in checkpoint:
        state = checkpoint["ema_state_dict"]
    else:
        state = checkpoint.get("model", checkpoint)
    state = copy.copy(state)
    if hasattr(state, "_metadata"):
        state._metadata = state._metadata.copy()
    consume_prefix_in_state_dict_if_present(state, "module.")
    return get_model(model).load_state_dict(state)

def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema, save_every_epoch=200, core=None):
    """
    save the model to path
    """
    save_ckpt = {
        'epoch': epoch + 1, 
        'model': model.state_dict(), 
        'ema_state_dict': ema.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'schedule': scheduler.state_dict(), 
        'loss': train_loss,
        'wandb_id': wandb_id
    }
    # Auxiliary modules (e.g., value_net) are stored separately so planner inference checkpoints remain compatible.
    if core is not None and hasattr(core, "get_aux_state_dict"):
        aux = core.get_aux_state_dict()
        if aux:
            save_ckpt["aux_state_dict"] = aux
    
    torch.save(save_ckpt, f"{save_path}/latest.pth")

    if epoch+1 >= save_every_epoch:
        with open(f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth', "wb") as f:
            torch.save(save_ckpt, f)

def load_model(path: str):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, weights_only=True, map_location="cpu")

    return ckpt


def resume_model(path: str, model, optimizer, scheduler, ema, device, core=None):
    """
    Resume from a checkpoint.

    `path` can be either:
      - a run directory containing `latest.pth`, or
      - a direct path to a `.pth` checkpoint file (e.g. `model_epoch_9_*.pth`).
    """
    ckpt_path = path
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "latest.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Stage on CPU: rank-zero checkpoints must not allocate a full copy on
    # cuda:0 in every worker. State loading moves tensors to each target model.
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")

    # load model               
    load_model_weights(model, ckpt, use_ema=False)
    print("Model load done")
    
    # load optimizer
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    else:
        print("no pretrained optimizer found; keeping initialized optimizer")
            
    # load schedule
    if "schedule" in ckpt:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    else:
        print("no schedule found; keeping initialized scheduler")
    
    # load step
    init_epoch = ckpt.get('epoch', 0)
    

    # Load wandb id
    wandb_id = ckpt.get('wandb_id')

    if "ema_state_dict" in ckpt:
        load_model_weights(ema.ema, ckpt['ema_state_dict'], use_ema=False)
        print("ema load done")
    else:
        # EMA is created before resume in the trainers. Without a saved shadow,
        # initialize it from the restored weights, not the pre-resume model.
        load_model_weights(ema.ema, get_model(model).state_dict(), use_ema=False)
        print("no ema shadow found; initialized from restored model")
    ema.ema.eval()
    for p in ema.ema.parameters():
        p.requires_grad_(False)

    # A present but incompatible auxiliary state must not silently reset the critic.
    if core is not None and "aux_state_dict" in ckpt and hasattr(core, "load_aux_state_dict"):
        core.load_aux_state_dict(ckpt["aux_state_dict"])
        print("aux state load done")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
