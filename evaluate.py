from pathlib import Path
import copy
from typing import NamedTuple
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from tqdm import tqdm
from the_well.benchmark.metrics import VRMSE

from util.gpustats import my_cuda_init
from read_global_params import GlobalParams
from datasets import validation_loader
from models import load_FNO
from transforms import Preprocessor



def validstats_summary(validstats_array):
    """Computes the mean and standard deviation of the validation VRMSE for 
    rho, P, T, v from a validation stats array saved during training.
    """

    vrmses = validstats_array

    vrmses_v_all = (vrmses[:,3] + vrmses[:,4] + vrmses[:,5]) / 3.

    avg_rho  = np.average(np.log10(vrmses[:,0]))
    avg_pres = np.average(np.log10(vrmses[:,1]))
    avg_temp = np.average(np.log10(vrmses[:,2]))
    avg_v    = np.average(np.log10(vrmses_v_all))

    std_rho  = np.std(np.log10(vrmses[:,0]))
    std_pres = np.std(np.log10(vrmses[:,1]))
    std_temp = np.std(np.log10(vrmses[:,2]))
    std_v    = np.std(np.log10(vrmses_v_all))

    avgs = [avg_rho, avg_pres, avg_temp, avg_v]
    stds = [std_rho, std_pres, std_temp, std_v]

    return avgs, stds


def select_best_epoch(model_name):
    """Selects the best epoch of a run using the mininum VRMSE on velocity."""

    gp = GlobalParams(model_name)    

    err_vels = []
    for epoch in gp.bench_list:

        vrmses = np.loadtxt(gp.get_fno_validstats_path(epoch))
        avgs, stds = validstats_summary(vrmses)
        err_vels.append(avgs[3]) #velocity error

    ind_epoch = np.argmin(err_vels)
    best_epoch = gp.bench_list[ind_epoch]

    return best_epoch


class EvalStats(NamedTuple):

    epochs: list
    density_av: np.ndarray
    density_sd: np.ndarray
    pressure_av: np.ndarray
    pressure_sd: np.ndarray
    temperature_av: np.ndarray
    temperature_sd: np.ndarray
    velocity_av: np.ndarray
    velocity_sd: np.ndarray

def calculate_evaluation_graph(model_name):
    """Returns an EvalStats class, which gives the validation 
    VRMSE performance (of rho, P, T, v) as a function of epoch.
    """

    gp = GlobalParams(model_name)

    avgs_all = []
    stds_all = []
    for epoch in gp.valid_list:

        vrmses = np.loadtxt(gp.get_fno_validstats_path(epoch))
        avgs, stds = validstats_summary(vrmses)
        avgs_all.append(avgs)
        stds_all.append(stds)

    avgs_all = np.array(avgs_all)
    stds_all = np.array(stds_all)

    return EvalStats(
        epochs = gp.valid_list,
        density_av  = avgs_all[:,0],
        density_sd  = stds_all[:,0],
        pressure_av = avgs_all[:,1],
        pressure_sd = stds_all[:,1],
        temperature_av = avgs_all[:,2],
        temperature_sd = stds_all[:,2],
        velocity_av    = avgs_all[:,3],
        velocity_sd    = stds_all[:,3]
    )


def plot_evaluation_graph(model_name):
    """Produces a plot showing the valid VRMSE as a function of
    epoch for each quantity rho, P, T, v, via an EvalStats class.
    """

    gp = GlobalParams(model_name)

    fig, axs = plt.subplots(2,2, figsize=(11,6))

    estats = calculate_evaluation_graph(model_name)
    axs[0,0].errorbar(estats.epochs, estats.density_av, yerr = estats.density_sd,
                    alpha=0.5, fmt='o'
    )
    axs[0,1].errorbar(estats.epochs, estats.pressure_av, yerr = estats.pressure_sd,
                    alpha=0.5, fmt='o'
    )
    axs[1,0].errorbar(estats.epochs, estats.temperature_av, yerr = estats.temperature_sd,
                    alpha=0.5, fmt='o'
    )
    axs[1,1].errorbar(estats.epochs, estats.velocity_av, yerr = estats.velocity_sd,
                    label=model_name, alpha=0.5, fmt='o'
    )

    axs[0,0].set_title('Density')
    axs[0,1].set_title('Pressure')
    axs[1,0].set_title('Temperature')
    axs[1,1].set_title('Velocity (combined)')

    fig.supylabel("log10 valid VRMSE")
    fig.savefig(f"{gp.paths.plots}/fno_evals.png")


def save_selected_model(model_name):
    """Saves the best benchmark (selected by lowest velocity error) model as a .pkl."""

    gp = GlobalParams(model_name)
    epoch = select_best_epoch(model_name)
    fno = load_FNO(gp, 'cpu', epoch)
    
    modelpath = f"{gp.paths.models}/best_fno.pkl"
    with open(modelpath, 'wb') as f:
        pickle.dump(fno, f)



def validate_fno_full(model_name):
    """Produces validation statistics for every validation trajectory (not just the subsample used during the training loop)."""

    epoch = select_best_epoch(model_name)

    gp = GlobalParams(model_name)
    gp_temp = copy.deepcopy(gp)
    gp_temp.validation.freq = 1

    device = my_cuda_init()

    fno = load_FNO(gp_temp, device, epoch)
    valid_loader, vmeta = validation_loader(gp_temp)
    pp = Preprocessor(gp_temp ,device)

    print("Now Validating Epoch No. {}".format(epoch))

    eval_results_all = []

    for _, batch in enumerate(bar := tqdm(valid_loader)):
        
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

            fx = fno(x) # requires B tF as first field for input, also outputs B tF

            fx = rearrange(fx, "B F Lx Ly Lz -> B 1 Lx Ly Lz F")
            y  = rearrange(y, "B F Lx Ly Lz -> B 1 Lx Ly Lz F")        

            bar.set_postfix()

        for _, (fx_i, y_i) in enumerate(zip(fx,y)):

            vrmse = VRMSE.eval( fx_i, y_i, meta=vmeta )[0] # requires F as last field
            eval_results_all.append(vrmse)

    eval_results_all = torch.stack(eval_results_all)
    evalnpy = eval_results_all.cpu().numpy()

    np.savetxt(f"{gp.paths.models}/best_eval_full.txt", evalnpy)
    return evalnpy


### --- ##### --- ###


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-names", nargs="+")
    args = parser.parse_args()

    for model_name in args.run_names:

        plot_evaluation_graph(model_name)
        save_selected_model(model_name)
