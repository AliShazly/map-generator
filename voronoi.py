import numpy as np
from scipy.spatial import Voronoi
import sys
from PIL import Image
import matplotlib.pyplot as plt

class VoronoiDiagram:

    def __init__(self, num_points=100, dimensions = (None, None)):
        self.points = np.random.random((num_points, 2))
        self.bounding_region = [min(self.points[:, 0]), max(self.points[:, 0]), min(self.points[:, 1]), max(self.points[:, 1])]


    def generate_voronoi(self):
        # https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells
        eps = sys.float_info.epsilon
        self.vor = Voronoi(self.points)
        self.filtered_regions = []
        for region in self.vor.regions:
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = self.vor.vertices[index, 0]
                    y = self.vor.vertices[index, 1]
                    if not (self.bounding_region[0] - eps <= x and x <= self.bounding_region[1] + eps and
                            self.bounding_region[2] - eps <= y and y <= self.bounding_region[3] + eps):
                        flag = False
                        break
            if region != [] and flag:
                self.filtered_regions.append(region)
        return self.vor

    def _region_centroid(self, vertices):
        signed_area = 0
        C_x = 0
        C_y = 0
        for i in range(len(vertices)-1):
            step = (vertices[i, 0]*vertices[i+1, 1])-(vertices[i+1, 0]*vertices[i, 1])
            signed_area += step
            C_x += (vertices[i, 0] + vertices[i+1, 0])*step
            C_y += (vertices[i, 1] + vertices[i+1, 1])*step
        signed_area = 1/2*signed_area
        C_x = (1.0/(6.0*signed_area))*C_x
        C_y = (1.0/(6.0*signed_area))*C_y
        return np.array([[C_x, C_y]])

    def relax_points(self, iterations=1):
        # https://stackoverflow.com/questions/17637244/voronoi-and-lloyd-relaxation-using-python-scipy
        for i in range(iterations):
            centroids = []
            for region in self.filtered_regions:
                vertices = self.vor.vertices[region + [region[0]], :]
                centroid = self._region_centroid(vertices)
                centroids.append(list(centroid[0, :]))
            self.points = centroids
            self.generate_voronoi()
        return self.vor


def voronoi_finite_polygons_2d(vor):
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)

    radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot(vor, noise_arr, show_plot=True):
        '''Plots a voronoi diagram with colors representing land and sea
        Params:
            vor : Scipy voronoi object
            noise_arr : 2D numpy perlin noise array

        '''
        regions, vertices = voronoi_finite_polygons_2d(vor)
        for region in regions:
            p = vertices[region]
            polygon = vertices[region]

            polygon_no_neg = [i for i in polygon if i[0] >= 0 and i[1] >= 0]
            polygon_normalized = [i for i in polygon_no_neg if i[0] <= 1 and i[1] <= 1]

            
            coords = polygon_no_neg[0]
            coords *= noise_arr.shape[0]

            
            x = int(round(coords[0]))
            y = int(round(coords[1]))

            try:
                color = noise_arr[x][y]
            except IndexError:
                # print(f'IndexError coords: {x}, {y}')
                color = 0

            if color < 50:
                plt.fill(*zip(*p), 'b', alpha=1)
            else:
                plt.fill(*zip(*p), 'g', alpha=1)


        # plt.plot(points[:,0], points[:,1], 'ko')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.savefig('images/voro.png')
        if show_plot:
            plt.show()