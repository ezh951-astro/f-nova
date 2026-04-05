import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from einops import rearrange
from tqdm import tqdm
from the_well.benchmark.metrics import VRMSE
import wandb

import util.gpustats as gpustats
from datasets import fno_train_loader, validation_loader
from models import training_state_dict, load_FNO
from transforms import Preprocessor, Physics



def train_fno( gp, restart, default_epoch, device, interactive=True, use_wandb=False ):

    tp = gp.training

    pp = Preprocessor(gp,device)
    phys = Physics(gp,device)

    # Training

    model, start_epoch, optim_state, sched_state = load_FNO(gp, device, default_epoch, restart=restart, mode='training')

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=tp.learn_rate, 
        weight_decay=tp.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = tp.lr_decay_period,
        gamma = tp.lr_decay_gamma
    )

    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)

    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_model_params}")

    # Train

    train_loader = fno_train_loader(gp)
    valid_loader, vmeta = validation_loader(gp)

    lambda_idg = gp.physics.lambda_idg

    lconsv_mass = gp.physics.lambda_consv_mass
    lconsv_momentum = gp.physics.lambda_consv_momentum

    nF = gp.data_summary.n_fields

    if interactive:
        print("Interactive Mode")
        print('Max Epochs {}, Saving Every {}'.format(tp.max_epochs,tp.epoch_save) )

    for epoch in range( start_epoch + 1, tp.max_epochs ):

        if interactive:
            print("Now Training Epoch No. {}".format(epoch))
            iterated_train = enumerate(bar := tqdm(train_loader))
        else:
            iterated_train = enumerate(train_loader)

        for i_b, batch in iterated_train:

            optimizer.zero_grad(set_to_none=True)

            # Load Data

            x = batch["input_fields"]
            y = batch["output_fields"]

            x = x.to(device)
            y = y.to(device)

            x = pp.preprocess(x)
            y = pp.preprocess(y)

            x = rearrange(x, "B Ti Lx Ly Lz F -> B (Ti F) Lx Ly Lz")      
            y = rearrange(y, "B To Lx Ly Lz F -> B (To F) Lx Ly Lz")

            # Forward

            fxiter = x
            fx = torch.empty( tp.batch_size, 
                             0, 
                             gp.data_summary.box_size, 
                             gp.data_summary.box_size, 
                             gp.data_summary.box_size, 
                             device=device 
            )
            for _ in range(gp.autoregression.order):

                fxiter = model(fxiter)
                fx = torch.cat([ fx, fxiter ], dim=1) # add: adjustable lambdas

            # Losses

            mse = (fx - y).square().mean()

            xu  = pp.unnormalize(x)
            fxu = pp.unnormalize(fx[ :, :nF, :,:,: ])

            if epoch >= gp.physics.tune_start:

                idgloss = lambda_idg * phys.ideal_gas(fxu).square().mean()

                consvloss_mass = lconsv_mass * (
                    phys.sum_mass(fxu) - phys.sum_mass(xu)
                ).square().mean()
                consvloss_momentum = lconsv_momentum * (
                    phys.sum_momentum(fxu) - phys.sum_momentum(xu)
                ).square().mean()

                consvloss = consvloss_mass + consvloss_momentum

            else:

                idgloss = torch.zeros((), device=device)
                consvloss = torch.zeros((), device=device)

            loss = mse + idgloss + consvloss

            # Backward + Update Optimizer

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), tp.grad_max)

            optimizer.step()

            # Logging

            if use_wandb and i_b % gp.logging.steps_per_loss == 0:
                loss_dict = {
                    "train/batch/loss_all": np.log10(loss.item()),
                    "train/batch/loss_mse": np.log10(mse.item())
                }
                if epoch >= gp.physics.tune_start:
                    if lambda_idg != 0:
                        loss_dict.update({ 
                            "train/batch/loss_idg": np.log10(idgloss.item()) 
                        })
                    if lconsv_mass != 0:
                        loss_dict.update({ 
                            "train/batch/loss_consv_mass": np.log10(consvloss_mass.item()) 
                        })
                    if lconsv_momentum != 0:
                        loss_dict.update({ 
                            "train/batch/loss_consv_momentum": np.log10(consvloss_momentum.item()) 
                        })
                wandb.log(loss_dict)

            if use_wandb and i_b % gp.logging.steps_per_grad == 0:
                wandb.log({
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
                    "optimizer/grad_norm": grad_norm
                })

            if interactive:
                bar.set_postfix(loss=loss.detach().item())

        if epoch % tp.epoch_save == 0:
            torch.save( training_state_dict( epoch, model, optimizer, scheduler ), gp.get_fno_bench_path(epoch) )

        scheduler.step()

        # Validation Loop

        if interactive:
            print("Now Validating Epoch No. {}".format(epoch))
            iterated_valid = enumerate(bar := tqdm(valid_loader))
        else:
            iterated_valid = enumerate(valid_loader)

        eval_results_all = []
        running_mse = 0

        for i_v, batch in iterated_valid:
            
            with torch.no_grad():

                x = batch["input_fields"]
                y = batch["output_fields"]

                x = x.to(device)
                y = y.to(device)

                x = pp.preprocess(x)
                y = pp.preprocess(y)

                x = rearrange(x, "B Ti Lx Ly Lz F -> Ti B F Lx Ly Lz")      
                y = rearrange(y, "B To Lx Ly Lz F -> To B F Lx Ly Lz")

                x = x[0]
                y = y[0] # B F Lx Ly Lz

                fx = model(x) # requires B tF as first field for input, also outputs B tF
                mse = (fx - y).square().mean()

                fx = rearrange(fx, "B F Lx Ly Lz -> B 1 Lx Ly Lz F")
                y  = rearrange(y, "B F Lx Ly Lz -> B 1 Lx Ly Lz F")

                running_mse += ( mse.item() - running_mse ) / ( i_v + 1 )            

            if interactive:
                bar.set_postfix(loss=running_mse)

            for _, (fx_i, y_i) in enumerate(zip(fx,y)):

                vrmse = VRMSE.eval( fx_i, y_i, meta=vmeta )[0] # requires F as last field
                eval_results_all.append(vrmse)

        eval_results_all = torch.stack(eval_results_all)

        evalmeans = eval_results_all.log10().mean(dim=0)
        evalstds = eval_results_all.log10().std(dim=0)

        if use_wandb:

            wandb.log({ "valid/loss_mse": np.log10(running_mse) })

            wandb.log({
                "valid/vrmse_rho": evalmeans[0].item(),
                "valid/vrmse_pressure": evalmeans[1].item(),                
                "valid/vrmse_temp": evalmeans[2].item(),
                "valid/vrmse_v": (evalmeans[3:6].mean()).item(),
            })

        evalnpy = eval_results_all.cpu().numpy()
        np.savetxt( gp.get_fno_validstats_path(epoch), evalnpy )