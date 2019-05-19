import numpy as np
import random
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


class MapGenerator:

    def _region_centroid(self, vertices):
        x_vals = [i[0] for i in vertices]
        y_vals = [i[1] for i in vertices]
        centroid_x = sum(x_vals) / len(x_vals)
        centroid_y = sum(y_vals) / len(y_vals)
        return centroid_x, centroid_y

    def _generate_base_terrain(self, noise_arr):
        land_polygons = []
        sea_polygons = []
        for region in self.regions:
            p = self.vertices[region]
            polygon = self.vertices[region]  # I have no clue why this is changing but it is

            polygon_normalized = [i for i in polygon if 0 <= i[0] <= 1 and 0 <= i[1] <= 1]

            coords = polygon_normalized[0]
            coords *= noise_arr.shape[0]

            x = int(round(coords[0]))
            y = int(round(coords[1]))

            try:
                color = noise_arr[x][y]
            except IndexError:
                color = 0

            if color < 50:
                sea_polygons.append(p)
                plt.fill(*zip(*p), 'b', alpha=1)
            else:
                land_polygons.append(p)
                plt.fill(*zip(*p), 'g', alpha=1)

        return land_polygons, sea_polygons

    def _generate_temperature(self, hot_point, cold_point):

        hot_centroid = self._region_centroid(hot_point)
        cold_centroid = self._region_centroid(cold_point)



        plt.fill(*zip(*hot_point), 'r', alpha=1)
        plt.fill(*zip(*cold_point), 'm', alpha=1)

        plt.plot([cold_centroid[0]], [cold_centroid[1]], marker='o', markersize=3, color="b")
        plt.plot([hot_centroid[0]], [hot_centroid[1]], marker='o', markersize=3, color="b")

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.savefig('images/voro.png', dpi=150)
        # plt.show()


    def generate_map(self, size=50, freq=30, lloyds=2, sigma=3.5):
        # make up data points
        size_sqrt = size
        size = size ** 2

        # compute Voronoi tesselation
        vor = VoronoiDiagram(size)
        vor.generate_voronoi()
        self.voronoi_diagram = vor.relax_points(lloyds)
        self.regions, self.vertices = voronoi_finite_polygons_2d(self.voronoi_diagram)

        # Get noise, turn into 1D array
        noise = Perlin(freq)
        noise_img = noise.create_image(save=True, width=200, height=200)
        noise_gauss = noise.gaussian_kernel(noise_img, nsig=sigma)
        noise_gauss.save('images/noise.png')
        noise_resized = noise_gauss.resize((size_sqrt, size_sqrt))
        noise_arr = np.array(noise_resized)

        land_polygons, sea_polygons = self._generate_base_terrain(noise_arr)

        hot_point, cold_point = random.sample(land_polygons, 2)
        self._generate_temperature(hot_point, cold_point)




if __name__ == '__main__':
    full_frame()
    generator = MapGenerator()
    generator.generate_map()