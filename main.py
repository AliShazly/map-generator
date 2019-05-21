import numpy as np
import random
import time
from functools import wraps
from PIL import Image
import matplotlib.pyplot as plt

from voronoi import *
from perlin_noise import Perlin

def full_frame(width=None, height=None):
    # https://gist.github.com/kylemcdonald/bedcc053db0e7843ef95c531957cb90f
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
    '''Prints the time a function took to run'''
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        elapsed = round(end - start, 2)
        print(f'{function.__name__} took {elapsed} seconds.')
        return result
    return wrapper


class MapGenerator:
    
    def _fill_polys(self, poly_list, color='r', alpha=1, single=False):
        '''Fills polygons on the pyplot graph'''
        # Setting the limits lower to hide map border
        plt.xlim(.05, .95)
        plt.ylim(.05, .95)
        if single:
            plt.fill(*zip(*poly_list), color, alpha=alpha)
        else:
            for i in poly_list:
                plt.fill(*zip(*i), color, alpha=alpha)

    def _distance(self, pt1, pt2):
        '''Returns the distance between two points'''
        return np.sqrt((pt2[0] - pt1[0])** 2 + (pt2[1] - pt1[1])** 2)

    def _polygon_is_in(self, poly, poly_lst):
        '''Checks if a polygon is in a list of polygons'''
        for i in poly_lst:
            if i == poly:
                return True
        return False

    def _get_centroid(self, polygon):
        '''Returns the centroid of a polygon'''
        x_vals = [i[0] for i in polygon]
        y_vals = [i[1] for i in polygon]
        centroid_x = sum(x_vals) / len(x_vals)
        centroid_y = sum(y_vals) / len(y_vals)
        return centroid_x, centroid_y

    def _get_polygon(self, centroid):
        '''Returns a polygon based on it/'s centroid'''
        index = self.centroids.index(centroid)
        return self.polygons[index]
    
    def _get_neighbors(self, main_polygon):
        '''Finds the adjacent polygons of a polygon'''
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

        # Removing duplicates
        neighbors = list(set([tuple([tuple(point) for point in polygon])
            for polygon in polygon_neighbors]))
        
        # Turning back into a nested list
        neighbors = [[list(point) for point in polygon] for polygon in neighbors]

        # Removing main polygon
        if self._polygon_is_in(main_polygon, neighbors):
            neighbors.remove(main_polygon)
        
        return neighbors

    def _generate_base_terrain(self, noise_arr):
        '''Generates the terrain as a binary map of land and ocean'''
        self.land_polygons = []
        self.water_polygons = []
        for polygon in self.polygons:
            # idk why i need to do this. self.polygons changes otherwise and i have no clue why
            p = polygon.copy()
            po = polygon.copy()

            polygon_stripped = [i for i in po if 0 <= i[0] <= 1
                                    and 0 <= i[1] <= 1]

            coords = self._get_centroid(polygon_stripped)
            coords = [i * noise_arr.shape[0] for i in coords]

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
        '''Changes edges of land masses into beaches'''
        self.beach_polys = []
        beach_indicies = []
        for idx, land_poly in enumerate(self.land_polygons):
            flag = False
            neighbors = self._get_neighbors(land_poly)
            for i in neighbors:
                if self._polygon_is_in(i, self.water_polygons):
                    flag = True
            if flag:
                self.beach_polys.append(land_poly)
                beach_indicies.append(idx)
        
        # Overwriting land cells
        for idx in sorted(beach_indicies, reverse=True):
            del self.land_polygons[idx]

    @timer
    def add_deep_water(self):
        '''Makes the ocean a deeper color than the lakes surrounded by land'''
        # Sorting to make sure the starting point is somewhere on the edge
        water_sorted = sorted(self.water_polygons, key=lambda x: x[0][0])
        self.deep_water_polys = []

        def check_validity(poly):
            if not self._polygon_is_in(poly, self.water_polygons):
                return False
            if self._polygon_is_in(poly, self.deep_water_polys):
                return False
            return True

        queue = [water_sorted[10]]
        while queue:
            poly = queue.pop()

            # Sometimes it iterates on duplicates, this fixes it.
            if not check_validity(poly):
                continue
            
            self.water_polygons.remove(poly)
            self.deep_water_polys.append(poly)
            neighbors = self._get_neighbors(poly)
                
            for i in neighbors:
                if check_validity(i):
                    queue.append(i)

    @timer
    def generate_map(self, size=75, freq=20, lloyds=2, sigma=3.15):
        '''Initializes the map and generates the base noise and voronoi diagram'''
        # make up data points
        size_sqrt = size
        size = size ** 2

        # compute Voronoi tesselation
        points = np.random.random((size, 2))
        vor = VoronoiDiagram(points)
        vor.generate_voronoi()
        self.voronoi_diagram = vor.relax_points(lloyds)
        regions, vertices = voronoi_finite_polygons_2d(self.voronoi_diagram)
        
        self.polygons = clip(points, regions, vertices)
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
    def plot(self, path='images/voro.png'):
        '''Plots and colors the voronoi map'''
        land = '#37754D'
        water_deep = '#285A8F'
        water_shallow = '#2685A6'
        beach = '#877965'
        self._fill_polys(self.polygons, 'black')
        self._fill_polys(self.land_polygons, land)
        self._fill_polys(self.water_polygons, water_shallow)
        self._fill_polys(self.beach_polys, beach)
        self._fill_polys(self.deep_water_polys, water_deep)

        plt.savefig(path, dpi=150)


if __name__ == '__main__':
    full_frame()
    generator = MapGenerator()
    generator.generate_map()
    generator.add_beaches()
    generator.add_deep_water()

    generator.plot()
