import numpy as np
from scipy.spatial import Voronoi
import sys
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon


class VoronoiDiagram:

    def __init__(self, points, dimensions = (None, None)):
        self.points = points
        self.bounding_region = [min(self.points[:, 0]), max(self.points[:, 0]), min(self.points[:, 1]), max(self.points[:, 1])]

    def _region_centroid(self, vertices):
        # I actually wrote this one! 
        x_vals = [i[0] for i in vertices]
        y_vals = [i[1] for i in vertices]
        centroid_x = sum(x_vals) / len(x_vals)
        centroid_y = sum(y_vals) / len(y_vals)
        return np.array([[centroid_x, centroid_y]])

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

    def relax_points(self, iterations=1):
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

def clip(points, regions, vertices):
    # https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells
    pts = MultiPoint([Point(i) for i in points])
    mask = pts.convex_hull
    new_vertices = []
    for region in regions:
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        poly = list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1]))
        poly = [list(i) for i in poly]
        new_vertices.append(poly)

    return new_vertices