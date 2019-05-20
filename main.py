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

    def _distance(self, pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0])** 2 + (pt2[1] - pt1[1])** 2)

    def _get_polygon(self, centroid):
        index = self.centroids.index(centroid)
        return self.polygons[index]
        
    def _get_neighbors(self, main_polygon):
        size = len(main_polygon)
        centroid = self._region_centroid(main_polygon)
        sorted_centroids = sorted(self.centroids, key=lambda x: self._distance(x, centroid))
        centroid_neighbors = sorted_centroids[:size]
        polygon_neighbors = [self._get_polygon(i) for i in centroid_neighbors]
        return polygon_neighbors

    def _generate_base_terrain(self, noise_arr):
        self.land_polygons = []
        self.sea_polygons = []

        for polygon in self.polygons:

            # idk why i need to do this. self.polygons changes otherwise and i have no clue why
            p = polygon.copy(order='C')
            po = polygon.copy(order='C')

            polygon_stripped = [i for i in po if 0 <= i[0] <= 1 and 0 <= i[1] <= 1]

            coords = polygon_stripped[0]
            coords *= noise_arr.shape[0]

            x = int(round(coords[0]))
            y = int(round(coords[1]))

            try:
                color = noise_arr[x][y]
            except IndexError:
                color = 0

            if color < 50:
                self.sea_polygons.append(p)
                plt.fill(*zip(*p), 'c', alpha=1)
            else:
                self.land_polygons.append(p)
                plt.fill(*zip(*p), 'g', alpha=1)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('images/voro.png')

    def _add_shallow_water(self):
        polys_sorted = sorted(self.sea_polygons, key=lambda x: x[0][0])
        land_sums = [np.sum(i) for i in self.land_polygons]

        x = polys_sorted[50]
        c = self._region_centroid(x)
        y = self._get_polygon(c)

        shallow_polys = []
        for p in polys_sorted:
            flag = False
            neighbors = self._get_neighbors(p)
            for i in neighbors:
                if np.sum(i) in land_sums:
                    flag = True
            if not flag:
                shallow_polys.append(p)
        
        for i in shallow_polys:
            plt.fill(*zip(*i), 'b', alpha=1)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('images/voro.png')

    def _generate_temperature_points(self, land_polygons, min_distance=0.1):
        hot_polygon, cold_polygon = random.sample(land_polygons, 2)
        
        hot_point = self._region_centroid(hot_polygon)
        cold_point = self._region_centroid(cold_polygon)

        # TODO: Do this differently, recursion breaks too easily
        if self._distance(hot_point, cold_point) < min_distance:
            return self._generate_temperature_points(land_polygons)

        plt.fill(*zip(*hot_polygon), 'r', alpha=1)
        plt.fill(*zip(*cold_polygon), 'p', alpha=1)

        plt.plot([cold_point[0]], [cold_point[1]], marker='o', markersize=3, color='b')
        plt.plot([hot_point[0]], [hot_point[1]], marker='o', markersize=3, color='b')

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.savefig('images/voro.png', dpi=150)
        # plt.show()

    def generate_map(self, size=50, freq=20, lloyds=2, sigma=3.5):
        # make up data points
        size_sqrt = size
        size = size ** 2

        # compute Voronoi tesselation
        vor = VoronoiDiagram(size)
        vor.generate_voronoi()
        self.voronoi_diagram = vor.relax_points(lloyds)
        regions, vertices = voronoi_finite_polygons_2d(self.voronoi_diagram)

        self.polygons = [vertices[region] for region in regions]
        self.centroids = [self._region_centroid(i) for i in self.polygons]

        # Get noise, turn into 1D array
        noise = Perlin(freq)
        noise_img = noise.create_image(save=True, width=200, height=200)
        noise_gauss = noise.gaussian_kernel(noise_img, nsig=sigma)
        noise_gauss.save('images/noise.png')
        noise_resized = noise_gauss.resize((size_sqrt, size_sqrt))
        noise_arr = np.array(noise_resized)

        self._generate_base_terrain(noise_arr)

        # self._generate_temperature_points(self.land_polygons)

        self._add_shallow_water()


if __name__ == '__main__':
    full_frame()
    generator = MapGenerator()
    generator.generate_map()