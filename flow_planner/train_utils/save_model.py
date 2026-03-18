import io
import torch
import os

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
    ckpt = torch.load(path, weights_only=True)

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

    ckpt = torch.load(ckpt_path, weights_only=True)

    # load model               
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model.load_state_dict({n.split("module.")[1]: v for n, v in ckpt.items()})
    print("Model load done")
    
    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")
            
    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")
    
    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0
    

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict({n: v for n, v in ckpt['ema_state_dict'].items()})
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    # load aux state (best-effort)
    if core is not None and "aux_state_dict" in ckpt and hasattr(core, "load_aux_state_dict"):
        try:
            core.load_aux_state_dict(ckpt["aux_state_dict"])
            print("aux state load done")
        except Exception:
            print("aux state load failed (ignored)")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema