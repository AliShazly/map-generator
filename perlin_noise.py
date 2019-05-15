import numpy as np
from PIL import Image

def generate_gradient_table(length):
    low, high = -1, 1
    gradient_table = np.zeros(length, dtype=tuple)
    for i in range(length):
        vector = (high - low) * np.random.random_sample(2) + low
        gradient_table[i] = tuple(vector)
    return gradient_table

def dot_product(vec1, vec2):
    return (vec1[0] * vec2[0]) + (vec1[1] * vec2[1])

def to_uint8(old_val):
    old_max, old_min = 1, -1
    new_max, new_min = 255, 0
    new_val = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return round(new_val)

def f(t):
    return 6*t**5 - 15*t**4 + 10*t**3

