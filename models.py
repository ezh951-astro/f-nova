import warnings

import torch
import torch.nn as nn
from neuralop.models import FNO


def load_FNO( gp, device, epoch, restart=True, mode='evaluation' ):
    """Loads an FNO model from a GlobalParams object.

    If loaded in training mode, will include the epoch, model state dict, optimizer state dict, 
    and scheduler state dict. In evaluation mode, only the model state dict will be loaded.

    Args:
        gp: GlobalParams object, which contains the run name
        device: Device the model will be assigned to
        epoch: Epoch of the loaded model
        restart: If false, a fresh, untrained model is loaded. If true, model will be loaded from a 
            past benchmark. In evaluation mode, must be True as there is no reason to evaluate an untrained 
            model. Default True.
        mode: 'training' or 'evaluation'. Default 'evaluation'.

    Returns:
        mode='training': Tuple of (model, start_epoch, optimizer_state_dict, scheduler_state_dict).
            start_epoch is -1 if restart=False. The optimizer and scheduler dicts may be None for fresh 
            or old-format checkpoints.
        mode='evaluation': The loaded model.

    Note: will check for a key 'model_state_dict': If the key exists, a checkpoint with the
    current formatting version that includes the optimizer and scheduler states as well
    is loaded. Otherwise, assumes that the checkpoint is only the model state dict (old
    checkpoint version). Will warn if the an old checkpoint version is loaded.  
    """

    if mode not in ['training','evaluation']:
        raise ValueError("mode must be training or evaluation")
    if mode=='evaluation' and not restart:
        raise ValueError("restart must be True for evaluation mode; can't perform eval on an untrained model")

    model = FNO(
        n_modes = (gp.model.n_modes, gp.model.n_modes, gp.model.n_modes), # 3 entries for 3D
        in_channels  = gp.data_summary.n_fields, #1 * F,
        out_channels = gp.data_summary.n_fields, #1 * F,
        hidden_channels = gp.model.hidden_channels,
        n_layers = gp.model.n_layers,
        factorization = gp.model.factorization,
        rank = gp.model.factorize_rank
    ).to(device)

    if not restart:

        start_epoch = -1 # 

        optimizer_state_dict = None
        scheduler_state_dict = None

    else:

        ckpt = torch.load( gp.get_fno_bench_path(epoch), map_location=device, weights_only=False )

        if 'model_state_dict' in ckpt.keys():

            model.load_state_dict(ckpt['model_state_dict'])
            optimizer_state_dict = ckpt['optimizer_state_dict']
            scheduler_state_dict = ckpt['scheduler_state_dict']

            start_epoch = epoch
            if start_epoch != ckpt['epoch']:
                warnings.warn("Loaded epoch number is different from the one saved in the checkpoint! Investigate.")

        else:

            warnings.warn("Using outdated checkpoint format that includes only model state dict; in the future, please save optimizer and scheduler states as well")

            model.load_state_dict(ckpt)
            optimizer_state_dict = None
            scheduler_state_dict = None

            start_epoch = epoch

    if mode == 'training':
        return model, start_epoch, optimizer_state_dict, scheduler_state_dict
    elif mode == 'evaluation':
        return model



def training_state_dict( epoch, model, optimizer, scheduler ):

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }