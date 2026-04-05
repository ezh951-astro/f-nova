import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def create_cmap(cmapname, lo=4.0, hi=8.0, Nbins=20):

    levels = MaxNLocator(nbins = Nbins).tick_values(lo,hi)
    cmap = plt.get_cmap(cmapname)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    return { 'cmap':cmap, 'norm':norm }