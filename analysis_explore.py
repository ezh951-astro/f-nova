import numpy as np
import matplotlib.pyplot as plt

import util.colormap_helper as cmh
from read_global_params import GlobalParams
import analysis_funcs as anf



def analyze(model_name, split, trajs, times):

    gp = GlobalParams(model_name)

    # Conservation of mass and momentum

    masses, momentums = anf.calculate_conservation_on_rollout(model_name, split, trajs, 0, 58)

    for traj in trajs:

        mass = masses[traj].cpu()
        momentum = momentums[traj].cpu()

        mass_norm = (mass - mass[0]) / mass[0]
        px_norm = (momentum[:,0] - momentum[0,0]) / momentum[0,0]
        py_norm = (momentum[:,1] - momentum[0,1]) / momentum[0,1]
        pz_norm = (momentum[:,2] - momentum[0,2]) / momentum[0,2]

        figCons, axsCons = plt.subplots(1,1, figsize=(7,6))

        axsCons.plot(mass_norm, label='mass')
        axsCons.plot(px_norm, label='px')
        axsCons.plot(py_norm, label='py')
        axsCons.plot(pz_norm, label='pz')

        axsCons.set_xlabel("Frame")
        axsCons.set_ylabel("Error on Quantity")

        axsCons.legend(loc='upper left')

        figCons.savefig(f"{gp.paths.plots}/conservation_{traj}.png")
        plt.close(figCons)


    # Phase diagram

    phases = anf.phase_distribution(model_name, split, trajs, thinning=99)

    for traj in trajs:
        for time in times:

            phase = phases[traj].cpu()

            figPh = plt.figure()
            axPh = figPh.add_subplot(111, projection='3d')

            axPh.scatter( phase[0,0,time], phase[0,1,time], phase[0,2,time],
                        s=1, alpha=0.4, label='pred' 
            )
            axPh.scatter( phase[1,0,time], phase[1,1,time], phase[1,2,time],
                        s=1, alpha=0.4, label='true' 
            )

            axPh.set_xlabel('Density')
            axPh.set_ylabel('Temperature')
            axPh.set_zlabel('Velocity')

            figPh.savefig(f"{gp.paths.plots}/phase_{traj}_{time}.png")
            plt.close(figPh)


    # Visualization

    # Colormaps
    field_min = [ 0.0, -5, 1, 0, 0, 0 ]
    field_max = [ 2.5, 2, 6, 40, 40, 40 ]
    cmkws = [ cmh.create_cmap("RdBu_r", lo=field_min[k], hi=field_max[k]) for k in range(6) ]

    for traj in trajs:

        dens_pred, dens_true = anf.show_rollout(model_name, split, traj, 0)
        temp_pred, temp_true = anf.show_rollout(model_name, split, traj, 2)
        velz_pred, velz_true = anf.show_rollout(model_name, split, traj, 5)

        dens_pred = dens_pred.sum(dim=3).log10().cpu().numpy()
        dens_true = dens_true.sum(dim=3).log10().cpu().numpy()

        temp_pred = temp_pred.log10().mean(dim=3).cpu().numpy()
        temp_true = temp_true.log10().mean(dim=3).cpu().numpy()

        velz_pred = velz_pred.std(dim=3).cpu().numpy()
        velz_true = velz_true.std(dim=3).cpu().numpy()

        figD, axsD = plt.subplots(2, len(times), figsize=(8,4))
        figT, axsT = plt.subplots(2, len(times), figsize=(8,4))
        figV, axsV = plt.subplots(2, len(times), figsize=(8,4))

        for i_t,time in enumerate(times):

            axsD[0,i_t].imshow(dens_pred[time], **(cmkws[0]))
            axsD[1,i_t].imshow(dens_true[time], **(cmkws[0]))
            axsD[0,i_t].set_title(f"t = {time}")

            axsT[0,i_t].imshow(temp_pred[time], **(cmkws[2]))
            axsT[1,i_t].imshow(temp_true[time], **(cmkws[2]))            
            axsT[0,i_t].set_title(f"t = {time}")

            axsV[0,i_t].imshow(velz_pred[time], **(cmkws[5]))
            axsV[1,i_t].imshow(velz_true[time], **(cmkws[5]))
            axsV[0,i_t].set_title(f"t = {time}")

        axsD[0,0].set_ylabel("Predicted")
        axsD[1,0].set_ylabel("True")
        figD.supylabel("Density")

        axsT[0,0].set_ylabel("Predicted")
        axsT[1,0].set_ylabel("True")
        figT.supylabel("Temperature")

        axsV[0,0].set_ylabel("Predicted")
        axsV[1,0].set_ylabel("True")
        figV.supylabel("Velocity Dispersion")

        figD.savefig(f"{gp.paths.plots}/density_{traj}.png")
        figT.savefig(f"{gp.paths.plots}/temperature_{traj}.png")
        figV.savefig(f"{gp.paths.plots}/velocity_{traj}.png")

        plt.close(figD)
        plt.close(figT)
        plt.close(figV)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-names", nargs="+")
    args = parser.parse_args()

    trajs = [0, 1] # trajectory ID within analysis loader
    times = [5, 10, 20] # timestamps within eah trajectory

    for model_name in args.run_names:

        analyze(model_name, 'valid', trajs, times)