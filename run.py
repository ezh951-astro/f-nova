import argparse
import yaml
from pathlib import Path

import wandb

from util.gpustats import my_cuda_init
import util.dict_helper as dictutil

from read_global_params import GlobalParams
from train import train_fno

with open("configs/config.yaml") as f:
    CONFIGFILE = yaml.safe_load(f)

PARAM_PATH = "configs/params.yaml"

LOGGING_CONFIG = {
    "dev":    dict(interactive=True,  use_wandb_init=False, use_wandb_finish=False),
    "live":   dict(interactive=True,  use_wandb_init=True,  use_wandb_finish=True),
    "series": dict(interactive=False, use_wandb_init=True,  use_wandb_finish=True),
    "sweep":  dict(interactive=False, use_wandb_init=False, use_wandb_finish=True),
}

def main():

    args = handle_args()

    if args.mode == 'sweep':

        wandb.init( project = CONFIGFILE["names"]["wandb_project"] )  

        indiv_param_path = CONFIGFILE["paths"]["essential"]["indiv_params"]
        with open(f"{indiv_param_path}/{wandb.run.id}.yaml", "w") as f:
            yaml.dump( dictutil.unflatten(dict(wandb.config)), f )

    device = my_cuda_init(verbose=True)

    run( args.experiment_name, 
        device, 
        args.restart, 
        args.default_epoch, 
        args.data_path,
        **LOGGING_CONFIG[args.mode] 
    )



def handle_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', choices=list(LOGGING_CONFIG.keys()), default="dev",
                        help='Mode of the run. dev, live, series (series of slurm runs), and sweep (through wandb).'
                        )
    parser.add_argument('-r', '--experiment-name', 
                        help='name of path to save benchmarks, controls all parameters'
                        )
    parser.add_argument('-e', '--default-epoch', 
                        help='the epoch to be used in cases where it must be specified'
                        )
    parser.add_argument('-R', '--restart', action = 'store_true',
                        help = 'restart the process'
                        )
    parser.add_argument('-d', '--data-path',
                        help = 'path to the datasets, if overriding default'
                        )

    args = parser.parse_args()

    if args.experiment_name == None:
        args.experiment_name = "default"
    
    return args


def run( name, 
        device, 
        restart, 
        default_epoch, 
        data_path,
        interactive=True, use_wandb_init=False, use_wandb_finish=False ):

    AllParams = GlobalParams(name)

    if data_path is not None:
        AllParams.paths.base_data = data_path

    # dump restartfile
    with open(Path(AllParams.paths.models) / "params.yaml", "w") as f:
        yaml.dump(AllParams.config_dicts.all, f)

    if use_wandb_init:
        wandb.init(
            project = CONFIGFILE["names"]["wandb_project"],
            name = AllParams.experiment_name,
            config = AllParams.config_dicts.run
        )

    use_wandb = use_wandb_init or use_wandb_finish

    if restart and default_epoch is None:
        default_epoch = AllParams.last_bench

    try:
        train_fno( AllParams, restart, default_epoch, device, interactive=interactive, use_wandb=use_wandb )
    finally:
        if use_wandb_finish:
            wandb.finish()

### ### ###

if __name__ == '__main__':
    main()
