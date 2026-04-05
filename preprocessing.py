from pathlib import Path

import torch
from the_well.utils.download import well_download

from util.gpustats import my_cuda_init
from util.transform_helper import pick_field, slice_by_field
from read_global_params import GlobalParams
from datasets import preprocessing_data_loader



def download_data(gp, overwrite = False, splits = ['train', 'valid', 'test']):

    def _download_split(split_name):

        pth = Path(gp.paths.base_data) / Path(gp.data_summary.dataset_name) / 'data' / Path(split_name)
        if overwrite or (not pth.exists()):
            well_download(base_path=gp.paths.base_data, dataset=gp.data_summary.dataset_name, split=split_name)

    for split_name in splits:
        print(f"Now downloading {split_name}")
        _download_split(split_name)

def extract_times(gp):

    timepath = f"{gp.paths.times}/times.pt"

    if Path(timepath).exists():

        times = torch.load( timepath )

    else:

        import os
        import h5py

        datapath = f"{gp.paths.base_data}/{gp.data_summary.dataset_name}/data/train"
        times = torch.zeros(gp.data_summary.n_times)

        for dir in os.listdir(datapath):

            print(f"{datapath}/{dir}")
            f = h5py.File(f"{datapath}/{dir}")
            loaded_times = torch.tensor(f[u"dimensions"][u"time"])

            diff = torch.abs(loaded_times - times)

            if diff.max() > 1.e-12:
                print("List of times inconsistent - saving new copy")
                times = loaded_times
                torch.save( loaded_times, timepath )
            else:
                print("List of times is consistent")

    return times

def extract_datastats(gp):

    nF = gp.data_summary.n_fields
    device = my_cuda_init()

    if Path(gp.paths.mu).exists() and Path(gp.paths.sigma).exists() and Path(gp.paths.ideal_constant).exists():

        mu    = torch.load(gp.paths.mu, map_location=device)
        sigma = torch.load(gp.paths.sigma, map_location=device)

        idgconst = torch.load(gp.paths.ideal_constant, map_location=device)

    else:

        preprocess_loader = preprocessing_data_loader(gp)

        with torch.no_grad():

            idealCs = []
            xs = []
            for i_b, batch in enumerate(preprocess_loader):

                print(f'Now preprocessing batch {i_b}')

                xb = batch["input_fields"]
                assert not (xb==0).any(), "Zeros in input data. Check if the right tensor is loaded; the data is known to not have zeros"

                if gp.preprocess.log:
                    xb[slice_by_field(xb,0,3,nF)] = torch.log10(xb[slice_by_field(xb,0,3,nF)])
                if gp.preprocess.asinh:
                    xb[slice_by_field(xb,3,6,nF)] = torch.asinh(xb[slice_by_field(xb,3,6,nF)])

                if gp.preprocess.log:
                    idealCb = xb[pick_field(xb,1,nF)] - xb[pick_field(xb,0,nF)] - xb[pick_field(xb,2,nF)]
                else:
                    idealCb = xb[pick_field(xb,1,nF)] / xb[pick_field(xb,0,nF)] / xb[pick_field(xb,2,nF)]

                for x, idealC in zip(xb,idealCb):
                    xs.append(x)
                    idealCs.append(idealC)

            xs = torch.stack(xs)
            idealCs = torch.stack(idealCs)

            mu    = xs.reshape(-1, nF).mean(dim=0)
            sigma = xs.reshape(-1, nF).std(dim=0)
            idgconst = idealCs.mean()

        torch.save( mu,    gp.paths.mu    )
        torch.save( sigma, gp.paths.sigma )
        torch.save( idgconst, gp.paths.ideal_constant )

    return mu, sigma, idgconst

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--name", required=True)
    args = parser.parse_args()

    gp = GlobalParams(args.name)

    extract_times(gp)
    extract_datastats(gp)