import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from voronoi import *
from perlin_noise import Perlin

def main():

    # make up data points
    size = 900
    size_sqrt = int(size ** (1/2))

    # compute Voronoi tesselation
    vor = VoronoiDiagram(size)
    vor.generate_voronoi()
    diagram = vor.relax_points(2)

    # Get noise, turn into 1D array
    noise = Perlin(100)
    noise_img = noise.create_image(save=True, width=500, height=500)
    noise_gauss = noise.gaussian_kernel(noise_img)
    noise_gauss.save('noise.png')
    noise_resized = noise_gauss.resize((size_sqrt, size_sqrt))
    noise_arr = np.array(noise_resized)

    plot(diagram, noise_arr)

if __name__ == '__main__':
    main()