import numpy
import random
from PIL import Image
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


def generate_points(width, num_points):
    return [[random.randint(0, width), random.randint(0, width)] for _ in range(num_points)]


if __name__ == '__main__':
    points = generate_points(500, 100)
    v = Voronoi(points)
    voronoi_plot_2d(v, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=0)
    plt.show()