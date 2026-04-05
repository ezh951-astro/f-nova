import os
import re
import yaml
import copy
from pathlib import Path
from types import SimpleNamespace
import warnings

from util.transform_helper import check_axis_shape_conflict
from util.dict_helper import extract_dotted_keys, merge_config_dict, load_yaml_required
    
###

with open("configs/config.yaml") as f: # this should be OK to load at start; CONFIG should not be touched, only PARAMS
    CONFIGFILE = yaml.safe_load(f)

PARAM_PATH = "configs/params.yaml"

###

class Config: # to turn a yaml-loaded dictionary into a class
    """
    Wraps a nested dictionary (e.g. from YAML) as an attribute-accessible object,
    so that dot access is possible rather than bracketing key strings.

    args:
        data (dict): A flat or nested dictionary to convert.

    example:
        cfg = Config({"model": {"learn_rate": 1e-3, "layers": 4}})
        cfg.model.learn_rate   # 1e-3
    """
    def __init__(self,data):

        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            elif isinstance(value, list):
                value = [
                    Config(v) if isinstance(v, dict) else v
                    for v in value
                ]
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

class GlobalParams(Config):
    """Unified configuration object for FNO experiment run.

    Loads and merges parameters from up to three sources in increasing priority:
        1. The default block for `model_type` in the global param file
        2. The run-specific block (from the global param file or an individual YAML)
        3. A restart file from a previous run (if one exists)

    After merging, the config is augmented with resolved filesystem paths
    (all created on init), data summary fields, and derived values like
    `preprocess.log_asinh_mode`. The result is accessible via dot notation
    through the parent `Config` class.

    Parameters:
        experiment_name : str
            Unique identifier for the run. Used to locate config files and
            construct all output paths.

    Attributes:
        In addition to attributes in the params.yaml and configs.yaml,
        restarted : bool
            True if a restart file was found and merged.
        bench_list : list of int
            Sorted epoch numbers for which a benchmark checkpoint exists.
        valid_list : list of int
            Sorted epoch numbers for which a validation stats file exists.
        last_bench : int
            Most recent benchmark epoch, or 0 if none exist.
        last_valid : int
            Most recent validation epoch, or 0 if none exist.
    """
    def __init__(self, experiment_name):

        self.experiment_name = experiment_name

        paramfile = load_yaml_required(PARAM_PATH)

        base_config = paramfile["fno"]["default"]

        run_config_exists = False
        try:
            run_config = paramfile["fno"][experiment_name]
            run_config_exists = True
        except KeyError:
            try:
                indiv_paramfile_dir = CONFIGFILE["paths"]["essential"]["indiv_params"]
                run_config = load_yaml_required(Path(indiv_paramfile_dir)/f"{experiment_name}.yaml")
                run_config_exists = True
            except FileNotFoundError:

                warnings.warn("Could not find params in either default or individualized paramfiles; checking for restartfile")
                run_config = {}

        self.config_dicts = SimpleNamespace()

        self.config_dicts.run = copy.deepcopy(run_config)

        # - Merge run-specific config with the default; run-specific overrides default - #
        config_data = merge_config_dict(base_config, run_config)


        # - If run exists and is being restarted, have any existing params override - #
        restart_config_path = f"{CONFIGFILE['paths']['by_run']['models']}/{experiment_name}/params.yaml"

        if Path(restart_config_path).exists():

            self.restarted = True
            restart_config = load_yaml_required(restart_config_path)
            config_data = merge_config_dict(config_data, restart_config)

        else:

            self.restarted = False
            if not run_config_exists:
                raise ValueError(f"Paramfile could not be located for run {experiment_name}")
    
        self.config_dicts.all = copy.deepcopy(config_data)


        # --- Augment config with paths and data summary (static values) --- #
        config_data.update({"data_summary": CONFIGFILE["data_summary"] })

        paths_global = CONFIGFILE["paths"]["global"]
        paths_local = { k: f"{v}/{experiment_name}" for k,v in CONFIGFILE["paths"]["by_run"].items() }
        paths_all = { **paths_global, **paths_local }

        for _,val in paths_all.items():
            Path( val ).mkdir(parents=True, exist_ok=True)

        config_data.update( { "paths": paths_all } )


        # --- Validate params --- #
        checked_params_list = ['training.batch_size', 'validation.batch_size', 'autoregression.order']

        checked_params = extract_dotted_keys(config_data, checked_params_list)
        forbidden_params = extract_dotted_keys(config_data, [ 'data_summary.n_fields', 'data_summary.box_size' ])
        check_axis_shape_conflict( checked_params, forbidden_params )


        # --- Convert dict to dot access --- #
        super().__init__(config_data)


        # --- Ensure critical attribute categories, which apply to both categories, exist --- #
        assert hasattr(self, 'preprocess'), "Config missing 'preprocess' block"
        assert hasattr(self, 'paths'), "Config missing 'paths' block"


        # --- Additional Paths outside of config_data --- #
        self.preprocess.log_asinh_mode = self.preprocess.log + 2*self.preprocess.asinh # No log no asinh 0 | yes log no asinh 1 | no log yes asinh 2 | yes log yes asinh 3

        self.paths.mu = "{}/mu_{}_{}_log{}.pt".format(
            self.paths.statistics,
            self.preprocess.norm_samples,
            self.preprocess.norm_cycle,
            self.preprocess.log_asinh_mode
        )
        self.paths.sigma = "{}/sigma_{}_{}_log{}.pt".format(
            self.paths.statistics,
            self.preprocess.norm_samples,
            self.preprocess.norm_cycle,
            self.preprocess.log_asinh_mode
        )
        self.paths.ideal_constant = "{}/idealc_{}_{}_log{}.pt".format(
            self.paths.statistics,
            self.preprocess.norm_samples,
            self.preprocess.norm_cycle,
            self.preprocess.log_asinh_mode
        )

        self.bench_list = self._get_numlist( self.paths.models, 'FNOBench_', '.pt' )
        self.valid_list = self._get_numlist( self.paths.models, 'validstats_', '.txt' )        

        self.last_bench = max( self.bench_list ) if len(self.bench_list) > 0 else 0
        self.last_valid = max( self.valid_list ) if len(self.valid_list) > 0 else 0

    def __repr__(self):

        return f"{self.__class__.__name__}({self.__dict__})"

    def _get_numlist(self, path, name_front, name_back):

        pattern = re.compile(r'^{}(\d+)\{}$'.format(name_front,name_back))

        nums = []
        for filename in os.listdir(path):
            m = pattern.match(filename)
            if m:
                nums.append( int(m.group(1)) )

        nums.sort()
        return nums
    
    def get_fno_bench_path(self,epoch):

        return f"{self.paths.models}/FNOBench_{epoch}.pt"

    def get_fno_validstats_path(self,epoch):

        return f"{self.paths.models}/validstats_{epoch}.txt"

    def get_training_residual_path(self,traj):
    
        Path(f'{self.paths.residuals}/train').mkdir(parents=True, exist_ok=True)
        return f"{self.paths.residuals}/train/{traj}.pt"

    def get_validation_residual_path(self, traj):
    
        Path(f'{self.paths.residuals}/valid').mkdir(parents=True, exist_ok=True)
        return f"{self.paths.residuals}/valid/{traj}.pt"

    def get_rollout_path(self, item, traj, start, stop, split='valid'):

        if item not in ['global_idx', 'fx', 'y', 'residual', 'vrmse']:
            raise ValueError("Allowed rollout items are global_idx, fx, y, residual, vrmse")

        Path(f'{self.paths.rollouts}/{split}').mkdir(parents=True, exist_ok=True)

        path_name = f"{self.paths.rollouts}/{split}/"
        path_name += item
        path_name += f"{traj}_{start}_{stop}"
        path_name += ".pt"

        return path_name