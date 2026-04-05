import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import time

import os

import util.submission_helper as submit

with open("configs/config.yaml") as f:
    CONFIGFILE = yaml.safe_load(f)

WORKING_DIR = os.getcwd()
EMAIL = CONFIGFILE["names"]["email"]

GRES = CONFIGFILE["compute"]["gres"]
CMEM = CONFIGFILE["compute"]["cmem"]
COMPUTE_HOURS = CONFIGFILE["compute"]["compute_hours"]

CONDA_ENV = CONFIGFILE["compute"]["conda_env"]

def slurm_run_series(run_list, copy_data_to_scratch = False):

    for run_name in run_list:

        slurm_script = write_slurm_script( "series", 
                                          run_name = run_name, 
                                          sweep_id = None, 
                                          starting_epoch = None,
                                          copy_data_to_scratch = copy_data_to_scratch
        )

        subprocess.run(["sbatch", slurm_script], check=True)

        submit.ensure_queue_submission(run_name)

        time.sleep(4)

def slurm_run_sweep(nruns, sweep_id, copy_data_to_scratch = False):

    for i in range(nruns):

        run_name = f"{sweep_id}_{datetime.now().strftime('%H%M%S')}_{i}"

        slurm_script = write_slurm_script( "sweep", 
                                          run_name = run_name,
                                          sweep_id = sweep_id,
                                          starting_epoch = None,
                                          copy_data_to_scratch = copy_data_to_scratch
        )

        subprocess.run(["sbatch", slurm_script], check=True)

        submit.ensure_queue_submission(run_name)

        time.sleep(4)

def slurm_run_indiv_custom(name, default_epoch, restart, copy_to_scratch=True):

    slurm_script = write_slurm_script( "live",
                                        run_name = name,
                                        starting_epoch = default_epoch,
                                        restart = restart,
                                        copy_data_to_scratch = copy_to_scratch
    )

    subprocess.run(["sbatch", slurm_script], check=True)

    submit.ensure_queue_submission(name)

    time.sleep(4)



def write_compute_settings(run_name,
                           gres = GRES,
                           cmem = CMEM,
                           compute_time = COMPUTE_HOURS):

    if compute_time <= 2:
        partition_name = 'short_gpu'
    else:
        partition_name = 'gpu'

    return f"""#SBATCH -J {run_name}
#SBATCH -p {partition_name}
#SBATCH --gres=gpu:{gres}
#SBATCH --exclude=gpu[01-06]
#SBATCH --mem={cmem}gb
#SBATCH --time={compute_time}:00:00
#SBATCH -o slurm/{run_name}.out
#SBATCH -e slurm/{run_name}.err
#SBATCH --mail-user={EMAIL}
#SBATCH --mail-type=ALL
"""

def write_dataset_copy_command():

    scratch_path = "$SCRATCH/datasets"

    default_data_path = CONFIGFILE["paths"]["global"]["base_data"]
    if not Path(default_data_path).exists():
        raise FileNotFoundError("Data path specified in Config could not be found.")

    data_origin_path = f"{WORKING_DIR}/{default_data_path}"

    dataset_name = CONFIGFILE["data_summary"]["dataset_name"]

    def _copy_data_item_to_scratch(item, src=data_origin_path, dest=scratch_path):

        cmd = f"mkdir -p {dest}/{item}\n"
        cmd += f"rsync -ah --progress {src}/{item}/. {dest}/{item}\n"
        cmd += f"ls {dest}/{item}\n"

        return cmd

    copy_command = f"echo 'Copying dataset to specified location...'\n"
    copy_command += _copy_data_item_to_scratch(f"{dataset_name}/data/train")
    copy_command += _copy_data_item_to_scratch(f"{dataset_name}/data/valid")
    copy_command += f"echo 'Done.'\n"

    return copy_command



def write_slurm_script( mode, 
                       run_name = None,
                       sweep_id = None,
                       starting_epoch = None,
                       restart = False,
                       copy_data_to_scratch = False,
):

    if run_name is None:
        raise ValueError("run_name is None; please specify name of the run.")

    if mode not in ['live', 'series', 'sweep']:
        raise ValueError(f"Slurm script submission not supported for mode {mode}. Must be series or sweep")
    
    if mode in ['live', 'series'] and sweep_id is not None:
        raise ValueError("Sweep ID not allowed for series run; must use sweep ID from wandb")


    # Handle starting epoch and restart

    if starting_epoch is None:
        starting_epoch_flag = ""
    else:
        assert isinstance(starting_epoch, int), "Starting Epoch must be Int"
        starting_epoch_flag = f"-e {starting_epoch}"

    if restart:
        restart_flag = "-R"
    else:
        restart_flag = ""


    # Generate the components of the file

    compute_settings = write_compute_settings(run_name)

    if copy_data_to_scratch:
        dataset_copy_command = write_dataset_copy_command()
        data_path_flag = f"-d $SCRATCH/datasets"
    else:
        dataset_copy_command = ""
        data_path_flag = ""


    # Generate the run command, with flags

    if mode in ['live', 'series']:
        flags = f"-m {mode} -r {run_name} {starting_epoch_flag} {restart_flag} {data_path_flag}"
        command = f"python run.py {flags}"

    elif mode == "sweep":
        #flags = f"-m {mode} -s {sweep_id}"
        wandb_project_id = f"{CONFIGFILE['names']['wandb_entity']}/{CONFIGFILE['names']['wandb_project']}/{sweep_id}"
        command = f"wandb agent --count 1 {wandb_project_id}" #must set flags in the wandb sweep yaml. #-- {flags}"


    # Write the file

    Path("slurm").mkdir(parents=True, exist_ok=True)
    slurm_script = f"slurm/torun_{run_name}.sh"
    Path(slurm_script).unlink(missing_ok=True)

    with open(slurm_script, "w") as f:
        f.write("#!/bin/bash -l\n")
        f.write(compute_settings)
        f.write("\nmodule load slurm\n")
        f.write(dataset_copy_command)
        f.write(f"\ncd {WORKING_DIR}\n")
        f.write("conda deactivate \nconda deactivate \nconda deactivate \n")
        f.write(f"conda activate {CONDA_ENV}\n\n")
        f.write("echo 'CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'\n")
        f.write("nvidia-smi\n\n")
        f.write(command)
        f.write("\n")

    submit.check_if_recently_generated(slurm_script)

    return slurm_script



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["series", "sweep", "live"], default="live")

    parser.add_argument("-r", "--run-names", nargs="+")

    parser.add_argument("-s", "--sweep-id")
    parser.add_argument("-n", "--nruns", type=int)

    parser.add_argument('-e', '--default-epoch', type=int,
                        help='the epoch to be used in cases where it must be specified'
                        )
    parser.add_argument('-R', '--restart', action = 'store_true',
                        help = 'restart the process'
                        )

    parser.add_argument('-c', '--copy-data-to-scratch', action='store_true', help='copies data to scratch before running')

    args = parser.parse_args()

    if args.mode == "series":
        if args.run_names is None:
            parser.error("-r/--run-names requires at least one run in series mode")
        if args.sweep_id is not None:
            parser.error("-s/--sweep-id is only valid for sweep mode")
        if args.nruns is not None:
            parser.error("-n/--nruns is only valid for sweep mode")

    if args.mode == "sweep":
        if args.run_names is not None:
            parser.error("-r/--run-names is only valid for series mode")
        if args.sweep_id is None:
            parser.error("-s/--sweep-id required for sweep mode")
        if args.nruns is None:
            args.nruns = 1
            print("Number of sweep runs to submit not specified; defaulting to 1.")



    if args.mode == "series":
        slurm_run_series(args.run_names, 
                         copy_data_to_scratch = args.copy_data_to_scratch)
        
    elif args.mode == "sweep":
        slurm_run_sweep(args.nruns, args.sweep_id, 
                        copy_data_to_scratch = args.copy_data_to_scratch)

    elif args.mode == "live":
        slurm_run_indiv_custom(args.run_names[0], 
                               default_epoch = args.default_epoch, 
                               restart = args.restart,
                               copy_to_scratch = args.copy_data_to_scratch)