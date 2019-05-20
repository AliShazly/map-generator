import numpy as np
import random
import time
from functools import wraps
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


def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        elapsed = round(end - start, 2)
        if elapsed > 0.0:
            print(f'{function.__name__} took {elapsed} seconds.')
        return result
    return wrapper


class MapGenerator:

    def _fill_polys(self, poly_list, color='r', alpha=1):
        for i in poly_list:
            plt.fill(*zip(*i), color, alpha=alpha)

    def _get_centroid(self, polygon):
        x_vals = [i[0] for i in polygon]
        y_vals = [i[1] for i in polygon]
        centroid_x = sum(x_vals) / len(x_vals)
        centroid_y = sum(y_vals) / len(y_vals)
        return centroid_x, centroid_y

    def _get_polygon(self, centroid):
        index = self.centroids.index(centroid)
        return self.polygons[index]

    def _distance(self, pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0])** 2 + (pt2[1] - pt1[1])** 2)
    
    def _get_neighbors(self, main_polygon):
        x, y = main_polygon[0][0], main_polygon[0][1]

        d_t = 0.1  # Distance threshold
        polygons_close = [i for i in self.polygons
            if x - d_t <= i[0][0] <= x + d_t
            and y - d_t <= i[0][1] <= y + d_t]
        
        polygon_neighbors = []
        for point in main_polygon:
            for polygon in polygons_close:
                if point in polygon:
                    polygon_neighbors.append(polygon)
        return polygon_neighbors

    @timer
    def _generate_base_terrain(self, noise_arr):
        self.land_polygons = []
        self.water_polygons = []

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
                self.water_polygons.append(p)
            else:
                self.land_polygons.append(p)

    @timer
    def add_beaches(self):
        # TODO: Do this differently. This is extremely dumb
        water_sums = [np.sum(i) for i in self.water_polygons]

        self.beach_polys = []
        beach_indicies = []
        for idx, land_poly in enumerate(self.land_polygons):
            flag = False
            neighbors = self._get_neighbors(land_poly)
            for i in neighbors:
                if np.sum(i) in water_sums:
                    flag = True
            if flag:
                self.beach_polys.append(land_poly)
                beach_indicies.append(idx)

        # Overwriting land cells
        for i in sorted(beach_indicies, reverse=True):
            del self.land_polygons[i]

    def generate_temperature_points(self, min_distance=0.1):
        hot_polygon, cold_polygon = random.sample(self.land_polygons, 2)
        
        hot_point = self._get_centroid(hot_polygon)
        cold_point = self._get_centroid(cold_polygon)

        # TODO: Do this differently, recursion breaks too easily
        if self._distance(hot_point, cold_point) < min_distance:
            return self._generate_temperature_points(self.land_polygons)
    
    @timer
    def generate_map(self, size=50, freq=20, lloyds=3, sigma=3.15):
        # make up data points
        size_sqrt = size
        size = size ** 2

        # compute Voronoi tesselation
        vor = VoronoiDiagram(size)
        vor.generate_voronoi()
        self.voronoi_diagram = vor.relax_points(lloyds)
        regions, vertices = voronoi_finite_polygons_2d(self.voronoi_diagram)

        self.polygons = [vertices[region] for region in regions]
        self.centroids = [self._get_centroid(i) for i in self.polygons]

        # Get noise, turn into 1D array
        noise = Perlin(freq)
        noise_img = noise.create_image(save=True, width=200, height=200)
        noise_gauss = noise.gaussian_kernel(noise_img, nsig=sigma)
        noise_gauss.save('images/noise.png')
        noise_resized = noise_gauss.resize((size_sqrt, size_sqrt))
        noise_arr = np.array(noise_resized)

        self._generate_base_terrain(noise_arr)
    
    @timer
    def plot(self):
        land = '#37754D'
        water = '#285A8F'
        beach = '#877965'
        self._fill_polys(self.land_polygons, land)
        self._fill_polys(self.water_polygons, water)
        self._fill_polys(self.beach_polys, beach)

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.savefig('images/voro.png', dpi=150)
        # plt.show()



if __name__ == '__main__':
    full_frame()
    generator = MapGenerator()
    generator.generate_map()
    generator.add_beaches()

    generator.plot()