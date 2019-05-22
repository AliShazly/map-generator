import numpy as np
import random
import time
from functools import wraps
import matplotlib.pyplot as plt

from voronoi import VoronoiDiagram, voronoi_finite_polygons_2d, clip
from perlin_noise import Perlin


def full_frame(width=None, height=None):
    # https://gist.github.com/kylemcdonald/bedcc053db0e7843ef95c531957cb90f
    '''Removes frame from matplotlib'''
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect(1)
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


class Polygon:
    def __init__(self, vertices, biome=None, coastline=False, elevation=None):
        self.vertices = vertices
        self.biome = biome
        self.coastline = coastline
        self.elevation = elevation

    def __getitem__(self, idx):
        return self.vertices[idx]

    def __setitem__(self, idx, val):
        self.vertices[idx] = val

    def __repr__(self):
        return f'{self.__class__.__name__}({self.vertices}, {self.biome}, {self.coastline}, {self.elevation})'

    @property
    def centroid(self):
        x_vals = [i[0] for i in self.vertices]
        y_vals = [i[1] for i in self.vertices]
        centroid_x = sum(x_vals) / len(x_vals)
        centroid_y = sum(y_vals) / len(y_vals)
        return centroid_x, centroid_y


class Biome:
    def __init__(self, color, group):
        self.color = color
        self.group = group

    def __repr__(self):
        return f'{self.__class__.__name__}({color}, {group})'


class MapGenerator:
    def __init__(self):
        self.land_beach = Biome('#A6977B', 'land')
        self.land_01 = Biome('#679459', 'land')
        self.land_02 = Biome('#85A979', 'land')
        self.land_03 = Biome('#9CB993', 'land')
        self.land_04 = Biome('#BCCFB5', 'land')
        self.shallow_water = Biome('#498DC9', 'water')
        self.deep_water = Biome('#356894', 'water')

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
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    def _get_closest_centroid(self, point, points):
        distances = [self._distance(point, i) for i in points]
        idx = distances.index(np.min(distances))
        return points[idx]

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

        # # Removing duplicates
        # neighbors = list(set([tuple([tuple(point) for point in polygon])
        #                       for polygon in polygon_neighbors]))

        # # Turning back into a nested list
        # neighbors = [[list(point) for point in polygon]
        #              for polygon in neighbors]

        try:
            polygon_neighbors.remove(main_polygon)
        except ValueError:
            pass

        return polygon_neighbors

    def _generate_base_terrain(self, noise_arr):
        '''Generates the terrain as a binary map of land and ocean'''
        for idx, polygon in enumerate(self.polygons):
            coords = polygon.centroid
            coords = [i * noise_arr.shape[0] for i in coords]

            x = int(round(coords[0]))
            y = int(round(coords[1]))

            try:
                color = noise_arr[x][y]
            except IndexError:
                color = 0

            if color < 50:
                self.polygons[idx].biome = self.shallow_water
            else:
                self.polygons[idx].biome = self.land_01

    @timer
    def _define_coastline(self):
        '''Makes a "coastline" that consists of water polys that border land polys'''
        for idx, poly in enumerate(self.polygons):
            if poly.biome.group == 'land':
                continue
            neighbors = self._get_neighbors(poly)
            for neighbor in neighbors:
                if neighbor.biome.group == 'land':
                    poly.coastline = True
                    self.polygons[idx] = poly
                    break

    @timer
    def add_deep_water(self):
        '''Makes the ocean a deeper color than the lakes surrounded by land'''

        def check_validity(poly):
            if not poly.biome == self.shallow_water:
                return False
            if poly.biome == self.deep_water:
                return False
            return True

        queue = []

        # TODO: Kinda dumb way to do this
        for poly in self.polygons:
            if poly.biome.group == 'water':
                queue.append(poly)
                break

        while queue:
            poly = queue.pop()

            # Sometimes it iterates on duplicates, this fixes it.
            if not check_validity(poly):
                continue

            idx = self.polygons.index(poly)
            self.polygons[idx].biome = self.deep_water
            neighbors = self._get_neighbors(poly)

            for i in neighbors:
                if check_validity(i):
                    queue.append(i)

    @timer
    def add_elevation(self):
        '''Defines a height from 0-1 for each land cell based on distance from water'''

        def normalize(val, mx, mn):
            return (val-mn) / (mx-mn)

        self._define_coastline()

        water_cents = [poly.centroid for poly in self.polygons
                       if poly.biome.group == 'water']

        distances = []
        for idx, poly in enumerate(self.polygons):
            if poly.biome.group == 'land':
                centroid = poly.centroid
                nearest_water = self._get_closest_centroid(
                    centroid, water_cents)
                distance = self._distance(centroid, nearest_water)
                distances.append((distance, idx))

        dist_max = max((i[0] for i in distances))
        dist_min = min((i[0] for i in distances))
        for d, idx in distances:
            elevation = normalize(d, dist_max, dist_min)
            self.polygons[idx].elevation = elevation

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

        polygons = sorted(clip(points, regions, vertices),
                          key=lambda x: x[0][0])
        polygons_stripped = [[vert for vert in poly if 0 <= vert[0] <= 1 and 0 <= vert[1] <= 1]
                             for poly in polygons]
        self.polygons = [Polygon(i) for i in polygons_stripped]

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

        def rgb2hex(rgb):
            rgb = tuple([int(i) for i in rgb])
            return '#%02x%02x%02x' % rgb

        # Giving the plot black borders
        self._fill_polys(self.polygons, 'black')

        for poly in self.polygons:
            if poly.biome.group == 'land':
                if 0 <= poly.elevation < .18:  # Too many beaches. lowered threshold
                    poly.biome = self.land_beach
                elif .18 <= poly.elevation < .4:
                    poly.biome = self.land_01
                elif .4 <= poly.elevation < .6:
                    poly.biome = self.land_02
                elif .6 <= poly.elevation < .8:
                    poly.biome = self.land_03
                else:
                    poly.biome = self.land_04
            self._fill_polys(poly, poly.biome.color, single=True)

        plt.savefig(path, dpi=300)


@timer
def main():
    full_frame(2, 2)
    generator = MapGenerator()
    generator.generate_map(size=50, freq=20, lloyds=2, sigma=3.15)
    generator.add_deep_water()
    generator.add_elevation()

    generator.plot()


if __name__ == '__main__':
    main()
