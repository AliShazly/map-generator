import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from voronoi import *
from perlin_noise import Perlin

# TODO: make this - https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

def main():
    from perlin_noise import Perlin

    # make up data points
    num_points = 8000
    # np.random.seed(1234)
    points = np.random.rand(num_points, 2)

    # compute Voronoi tesselation
    vor = VoronoiDiagram(num_points)
    vor.generate_voronoi()
    diagram = vor.relax_points(2)

    # Get noise, turn into 1D array
    noise = Perlin(50)
    noise_img = noise.create_image(save=True, width=500, height=500)
    noise_img = noise_img.resize((50, 50))
    noise_arr = np.array(noise_img).reshape(2500)


    # plot
    regions, vertices = voronoi_finite_polygons_2d(diagram)

    # colorize
    # for region, color in zip(regions, noise_arr):
    #     polygon = vertices[region]
    #     if color < 100:
    #         plt.fill(*zip(*polygon), 'b', alpha=1)
    #     else:
    #         plt.fill(*zip(*polygon), 'g', alpha=1)

    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=1)

    # plt.plot(points[:,0], points[:,1], 'ko')
    plt.xlim(.3, .8)
    plt.ylim(.3, .8)

    plt.savefig('voro.png')
    plt.show()

if __name__ == '__main__':
    main()