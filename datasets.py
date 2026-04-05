import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from the_well.data import WellDataset

def _load_well_dataset(gp, well_split_name):
    """Using a GlobalParams object, loads a split data into a WellDataset.

    Args:
        gp: GlobalParams object
        well_split_name: "train", "valid", or "test"

    Returns:
        tuple of (dataset, nSteps, nSets), where:
            dataset: dataset readable by a DataLoader.
            nSteps: Number of input-output steps, where multiple timesteps can live
                within an output field for autoregression order > 1.
            nSets: Number of trajectories.
    """

    dataset = WellDataset(
        well_base_path = gp.paths.base_data,
        well_dataset_name = gp.data_summary.dataset_name,
        well_split_name = well_split_name,
        n_steps_input = 1,
        n_steps_output = gp.autoregression.order,
        use_normalization = False
    )

    nSteps = gp.data_summary.n_times - gp.autoregression.order

    nSets_float = len(dataset) / (nSteps)
    nSets = int(nSets_float)
    assert nSets == nSets_float, "nSets doesn't match its integer value"

    return dataset, nSteps, nSets


def _get_indices(gp, well_split_name):
    """Using gp.base_split parameters, obtain indices to further split a dataset."""

    _, nsteps, nsets = _load_well_dataset(gp, well_split_name)

    split = int(gp.base_split.proportion * nsets)
    np.random.seed(gp.base_split.seed)
    idx = np.random.permutation(nsets)

    interest_idx_set   = idx[:split]
    holdout_idx_set    = idx[split:]

    interest_aux = [ [ iset*nsteps + t for t in range(nsteps) ] for iset in interest_idx_set ]
    holdout_aux = [ [ iset*nsteps + t for t in range(nsteps) ] for iset in holdout_idx_set ]

    interest_idx   = np.sort(np.concatenate(interest_aux)) if interest_aux else np.array([])
    holdout_idx    = np.sort(np.concatenate(holdout_aux)) if holdout_aux else np.array([])

    return interest_idx, holdout_idx



def fno_train_loader(gp):
    """Loads data for FNO training into a DataLoader."""

    trainset, nsteps, nsets_train = _load_well_dataset(gp,"train")
    train_idx, holdout_idx = _get_indices(gp,"train")

    fno_train_loader = DataLoader(
        dataset=Subset(trainset,train_idx),
        shuffle=True,
        batch_size=gp.training.batch_size,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    return fno_train_loader


def preprocessing_data_loader(gp):

    all_training_dataset, _, _ = _load_well_dataset(gp, "train")

    preprocessing_idx = range(0, gp.preprocess.norm_samples, gp.preprocess.norm_cycle)

    preprocess_loader = DataLoader(
        dataset = Subset(all_training_dataset, preprocessing_idx),
        shuffle = False,
        batch_size = gp.validation.batch_size, # uses this batch size as a general batch size for processing large amounts of gradient-free data
        num_workers = 1,
        drop_last = True,
        pin_memory = True,
        persistent_workers = True
    )

    return preprocess_loader


def validation_loader(gp, shift=0):
    """Loads data for validation during a training loop.
    
    Args:
        gp: GlobalParams object
        shift: Controls the subset of items sampled (i % gp.validation.freq = shift).

    Returns:
        validation_loader: DataLoader with sampled validation data.
        validset.metadata: metadata for using The Well VRMSE function.
    """

    validset, nsteps_valid, nsets_valid = _load_well_dataset(gp,"valid")

    validation_idx = range( shift, nsets_valid * nsteps_valid, gp.validation.freq )

    validation_loader = DataLoader(
        dataset=Subset(validset,validation_idx),
        shuffle=False, # shuffle = True, to allow for last batch to be more represented?
        batch_size=gp.validation.batch_size,
        num_workers=1,
        drop_last=True, # Drop last should be true so as to avoid an accidental batch size of nF=6
        pin_memory=True,
        persistent_workers=True
    )

    return validation_loader, validset.metadata



class AnalysisLoader:
    """Data object for post-hoc analysis.
    
    Individual input-output steps can be accessed via indices [i,j] where i is the trajectory ID within the split,
    and j is the step within the trajectory.

    Args:
        gp: GlobalParams object. Note: Deep-copied internally and modified
            (autoregression order forced to 1; base_split proportion
            set to 1.0 for valid/test splits), so the caller's
            object is not mutated.
        split_name: Which subset to load. One of:
            ``'fno_train'`` — training trajectories used for the base FNO.
            ``'residual_train'`` — training trajectories not used in the base FNO.
            ``'valid'`` — full validation set.
            ``'test'`` — full test set.
    """

    def __init__(self, gp, split_name):

        gp_anal = copy.deepcopy(gp)
        gp_anal.autoregression.order = 1

        if split_name in ['fno_train', 'residual_train']:
            well_split_name = 'train'
        elif split_name in ['valid', 'test']:
            gp_anal.base_split.proportion = 1.0
            well_split_name = split_name
        else:
            raise ValueError("analysis_loader split name must be: fno_train, residual_train, valid, test")

        index_set = 1 if split_name == 'residual_train' else 0

        dataset, nSteps, nSets = _load_well_dataset(gp_anal, well_split_name)
        self.dataset = dataset
        self.nSteps = nSteps
        self.nSets = nSets

        assert nSteps == gp_anal.data_summary.n_times - 1, \
            f"In analysis mode, nSteps must be {gp_anal.data_summary.n_times - 1}"

        self.idx = _get_indices(gp_anal, well_split_name)[index_set].reshape(-1, nSteps)
        sets_arr = self.idx // nSteps
        self.sets = sets_arr[:,0]

        for i, row in enumerate(sets_arr):
            assert all(x == row[0] for x in row), f"Row {i} has differing elements: {row}"

        self.metadata = dataset.metadata
        self.data_summary = gp.data_summary

    def __getitem__(self, ij):

        i, j = ij
        return self.dataset[self.idx[i, j]], self.idx[i,j]