import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from voronoi import *
from perlin_noise import Perlin

# https://gist.github.com/kylemcdonald/bedcc053db0e7843ef95c531957cb90f
def full_frame(width=None, height=None):
    '''Removes frame from matplotlib'''
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


def main(size=30, freq=100, lloyds=2):
    # make up data points
    size_sqrt = size
    size = size ** 2
    # size_sqrt = int(size ** (1/2))

    # compute Voronoi tesselation
    vor = VoronoiDiagram(size)
    vor.generate_voronoi()
    diagram = vor.relax_points(lloyds)

    # Get noise, turn into 1D array
    noise = Perlin(freq)
    noise_img = noise.create_image(save=True, width=200, height=200)
    noise_gauss = noise.gaussian_kernel(noise_img)
    noise_gauss.save('images/noise.png')
    noise_resized = noise_gauss.resize((size_sqrt, size_sqrt))
    noise_arr = np.array(noise_resized)

    plot(diagram, noise_arr, show_plot=False)

if __name__ == '__main__':
    full_frame()
    main()