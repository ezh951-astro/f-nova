from pathlib import Path
import warnings
from typing import NamedTuple

import numpy as np
import torch
from einops import rearrange

from the_well.benchmark.metrics import VRMSE

from util.gpustats import my_cuda_init
import util.transform_helper as tfh

from read_global_params import GlobalParams, CONFIGFILE
from datasets import AnalysisLoader
from models import load_FNO
from transforms import Preprocessor, Physics
from evaluate import select_best_epoch

NF      = CONFIGFILE["data_summary"]["n_fields"]
NTIMES  = CONFIGFILE["data_summary"]["n_times"]
BOXSIZE = CONFIGFILE["data_summary"]["box_size"]
BOXDIMS = CONFIGFILE["data_summary"]["box_dims"]



##### ----- ### --- Data Tensor Manipulation --- ### ----- #####

def get_field(x, *ij):

    if len(ij) == 2:
        i, j = ij
        result = x[ tfh.slice_by_field(x,i,j,NF) ]

    elif len(ij) == 1:
        i = ij[0]
        result = x[ tfh.pick_field(x,i,NF) ]

    else:
        raise ValueError(f"ij must be 1 or 2 values, got {len(ij)}")
    
    return result

def cell_distribution(x, thinning=1):

    axlist = [ i for i,s in enumerate(x.shape) if s==BOXSIZE ]
    assert all(b == a + 1 for a, b in zip(axlist, axlist[1:])), "axlist must be consecutive; please ensure Lx, Ly, Lz are consecutive axes."

    num_axes = len(axlist)
    resulting_len = BOXSIZE ** num_axes

    x = x.flatten(start_dim=axlist[0], end_dim=axlist[-1])

    cells_dim = tfh.find_field_axis(x.shape, resulting_len)
    x = torch.movedim(x, cells_dim, 0)
    x = x[::thinning]
    x = torch.movedim(x, 0, -1)

    return x




##### ----- ### --- Rollouts --- ### ----- #####

class Rollout(NamedTuple):
    global_idx: np.ndarray
    fx: torch.Tensor
    y: torch.Tensor
    residual: torch.Tensor
    vrmse: torch.Tensor

def rollout_one_trajectory(traj_id, start, stop, 
                           fno, analysis_loader, preprocessor):

    device = preprocessor.device

    with torch.no_grad():

        item_init,_ = analysis_loader[traj_id,start]

        x_init = item_init["input_fields"].to(device)
        x_init = rearrange(x_init, "B Lx Ly Lz F -> B F Lx Ly Lz")
        x_init = preprocessor.preprocess(x_init)

        fxiter = x_init

        fx = []
        y  = []

        yidxs = []

        for j_t in range(start,stop):

            fxiter = fno(fxiter)

            fx.append(fxiter)

            item, idx = analysis_loader[traj_id,j_t]
            yiter = item["output_fields"].to(device)
            yiter = rearrange(yiter, "B Lx Ly Lz F -> B F Lx Ly Lz")
            yiter = preprocessor.preprocess(yiter)

            y.append(yiter)
            yidxs.append(idx)

        fx = torch.stack(fx)
        y = torch.stack(y)

        fx = rearrange(fx, "T B F Lx Ly Lz -> (T B) Lx Ly Lz F")
        y = rearrange(y, "T B F Lx Ly Lz -> (T B) Lx Ly Lz F")

        vrmse = VRMSE.eval(fx, y, meta=analysis_loader.metadata)

        res = (fx - y).half()
        fx = fx.half()

        # first ensure that (T B) dimension does not interfere with F in automatic casting (postprocess):
        needs_pad = fx.shape[0] == analysis_loader.data_summary.n_fields
        if needs_pad:

            warnings.warn("Number of times matches the number of fields. Using workaround to avoid shape collision.")

            pad = torch.zeros_like(fx.narrow(0,0,1))
            fx = torch.cat([fx,pad], dim=0)
            y = torch.cat([y,pad], dim=0)
            #print(fx.shape,y.shape)

        fx = preprocessor.postprocess(fx) # must occur after VRMSE and residuals are calculated
        y = preprocessor.postprocess(y)

        if needs_pad:

            fx = fx[:-1]
            y = y[:-1]
            #print(fx.shape,y.shape)

    return Rollout(global_idx=np.array(yidxs), fx=fx, y=y, residual=res, vrmse=vrmse)

def rollout_general(model_name, 
                    start = 0, stop = NTIMES-1,
                    trajs = None, 
                    items = ["vrmse"],
                    split = "valid",
                    override = False,
                    special_epoch = None
):

    if not set(items) <= set(Rollout._fields):
        raise ValueError(f"Invalid items: {set(items) - set(Rollout._fields)}")

    gp = GlobalParams(model_name)
    device = my_cuda_init()

    analysis_loader = AnalysisLoader(gp, split)

    if special_epoch is None:
        epoch = select_best_epoch(model_name)
        print(f'Best epoch is {epoch}')
    else:
        epoch = special_epoch
    fno = load_FNO(gp, device, epoch) # default restart true and mode evaluation for instinctive loading

    pp = Preprocessor(gp, device)

    if trajs is None:
        trajs = analysis_loader.sets

    result = {}
    for traj in trajs:

        print(f'Trajectory No. {traj}')
        if traj not in analysis_loader.sets:
            warnings.warn(f"Trajectory {traj} not in {split} split, skipping.")
            continue

        needs_compute = override or any(
            not Path(gp.get_rollout_path(item, traj, start, stop, split=split)).exists()
            for item in items
        )

        result_of_traj = {}

        if needs_compute:

            rollout = rollout_one_trajectory(traj, start, stop, 
                                             fno, analysis_loader, pp)

            for item in items:
                itempath = gp.get_rollout_path(item, traj, start, stop, split=split)
                result_of_item = getattr(rollout, item)
                torch.save(result_of_item, itempath)
                result_of_traj.update({ item:result_of_item })
        
        else:

            for item in items:
                itempath = gp.get_rollout_path(item, traj, start, stop, split=split)
                result_of_item = torch.load(itempath)
                result_of_traj.update({ item:result_of_item })

        result.update({ traj:result_of_traj })

    return result

def calculate_indiv_training_residuals(model_name):

    gp = GlobalParams(model_name)
    device = my_cuda_init()

    analysis_loader = AnalysisLoader(gp,'residual_train')

    best_epoch = select_best_epoch(model_name)
    print(f'Best epoch is {best_epoch}')
    model = load_FNO(gp, device, best_epoch) # default restart true and mode evaluation for instinctive loading

    pp = Preprocessor(gp,device)
        
    for traj in analysis_loader.sets:

        print(f'Trajectory No. {traj}')

        res_all = []

        for time in range(NTIMES-1): # Varying times is why this is different from rollout_general

            rollout = rollout_one_trajectory(traj, time, time+1, model, analysis_loader, pp) # No corrector
            residual = rollout.residual[0]

            res_all.append(residual)

        res_all = torch.stack(res_all)

        torch.save(res_all, gp.get_training_residual_path(traj))

def calculate_indiv_validation_residuals(model_name):

    gp = GlobalParams(model_name)
    device = my_cuda_init()

    analysis_loader = AnalysisLoader(gp,'valid')

    best_epoch = select_best_epoch(model_name)
    print(f'Best epoch is {best_epoch}')
    model = load_FNO(gp, device, best_epoch) # default restart true and mode evaluation for instinctive loading

    pp = Preprocessor(gp,device)
        
    for traj in analysis_loader.sets:

        print(f'Trajectory No. {traj}')

        res_all = []

        for time in range(NTIMES-1): # Varying times is why this is different from rollout_general

            rollout = rollout_one_trajectory(traj, time, time+1, model, analysis_loader, pp) # No corrector
            residual = rollout.residual[0]

            res_all.append(residual)

        res_all = torch.stack(res_all)

        torch.save(res_all, gp.get_validation_residual_path(traj))




##### ----- ### --- Analysis Macros --- ### ----- #####

def show_rollout(model_name, split, traj, field):
    """
    3D time evolution map of a field over a rollout.

    Args:
        model_name: name of the FNO model
        split: name of the data split: fno_train, residual_train, valid, or test
        trajs: list of trajectory IDs to calculate for
        field: 0 to 5: density, pressure, temperature, vx, vy, vz

    Returns:
        pred, true: [T, X, Y, Z] spatiotemporal 3D evolution of the selected field.
    """
    rollout_items = rollout_general(model_name, items=['fx','y'], trajs=[traj], split=split)

    fx = rollout_items[traj]['fx']
    y  = rollout_items[traj]['y']

    pred = get_field(fx, field)
    true = get_field(y, field)

    return pred, true


def calculate_conservation_on_rollout(model_name, split, trajs, start, stop):
    """
    Finds the total mass and momentum evolution for a rollout.

    Args:
        model_name: name of the FNO model
        split: name of the data split: fno_train, residual_train, valid, or test
        trajs: list of trajectory IDs to calculate for
        start: start time (0 to 57)
        stop: stop time (1 to 58)

    Returns:
        mass: dict { traj: [nTimes] } where nTimes = stop - start
        momentum: dict { traj: [nTimes, 3] } for x, y, and z directions
    """

    gp = GlobalParams(model_name)
    device = my_cuda_init()

    phys = Physics(gp,device)

    rollout_items = rollout_general(model_name, 
                    trajs=trajs, start=start, stop=stop, 
                    items=['fx'], split=split)

    masses = {}
    momentums = {}
    for traj in trajs:

        fxs = rollout_items[traj]['fx']

        mass = phys.sum_mass(fxs, pretransformed=False)
        momentum = phys.sum_momentum(fxs, pretransformed=False)

        masses.update({ traj:mass })
        momentums.update({ traj:momentum })

    return masses, momentums


def phase_distribution(model_name, split, trajs, thinning=1):
    """
    Finds the phase distribution (density, temperature, velocity) for a rollout, across all times.

    Args:
        model_name: name of the FNO model
        split: name of the data split: fno_train, residual_train, valid, or test
        trajs: list of trajectory IDs to calculate for
        thinning: sampling frequency n, where every nth cell is included in the distribution; default 1

    Returns:
        phases: dict { traj: [pred/true (2), rho/temp/vel (3), nTimes, nCells] }
    """

    rollout_items = rollout_general(model_name, items=['fx','y'], trajs=trajs, split=split)
    
    phases = {}
    for traj in trajs:

        fx = rollout_items[traj]['fx']
        y  = rollout_items[traj]['y']

        rho_pred = get_field(fx, 0)
        temp_pred = get_field(fx, 2)
        vel2_pred = get_field(fx, 3).square() + get_field(fx, 4).square() + get_field(fx, 5).square()

        rho_true = get_field(y, 0)
        temp_true = get_field(y, 2)
        vel2_true = get_field(y, 3).square() + get_field(y, 4).square() + get_field(y, 5).square()

        phase_pred = torch.stack([rho_pred, temp_pred, vel2_pred])
        phase_true = torch.stack([rho_true, temp_true, vel2_true])

        phase = torch.stack([phase_pred, phase_true])
        phase = cell_distribution(phase, thinning=thinning)

        phase = phase.log10()
        phase[:,2] /= 2 # changes log(v^2) to log(|v|)

        phases.update({ traj:phase })

    return phases