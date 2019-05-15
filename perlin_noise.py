import math
import random
from PIL import Image
import hashlib

# http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
# https://eev.ee/blog/2016/05/29/perlin-noise/
# https://en.wikipedia.org/wiki/Perlin_noise

class Perlin:

    def __init__(self, grid, seed=None):
        self.grid = grid
        self.seed = list(range(256))
        if seed:
            random.seed(seed)
        random.shuffle(self.seed)
        self.gradient_vectors = [(-1.0, 1.0),
                                (math.sqrt(2), 0.0),
                                (1.0, -1.0),
                                (0.0, math.sqrt(2)),
                                (-math.sqrt(2), 0.0),
                                (1.0, 1.0),
                                (0.0 , -math.sqrt(2)),
                                (-1.0, -1.0)]

    def _lerp(self, first, second, by):
        '''Linear interpolation between two points'''
        return (first * by) + (second * (1 - by))

    def _f(self, t):
        '''Blending function'''
        return ((6 * (t**5)) - (15 * (t**4)) + (10 * (t**3)))

    def _dot(self, vec1, vec2):
        '''Calculates the dot product of two vectors'''
        return (vec1[0] * vec2[0]) + (vec1[1] * vec2[1])

    def _get_gradient(self, x, y):
        '''Picks a psuedo-random gradient from the gradient table'''
        string = str(x) + str(y)
        hashed = int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 256
        seed = self.seed[hashed]
        return self.gradient_vectors[seed % 8]
		
    def perlin_noise(self, x, y):
        '''Calculates perlin noise for a given 2D point. Returns a value between 0 and 1'''
        blockX = math.floor(x / self.grid)  # Calculates local grid points
        blockY = math.floor(y / self.grid)

        x = (x % self.grid) / float(self.grid)  # Normalize x and y to these grid points
        y = (y % self.grid) / float(self.grid)

        grad1 = self._get_gradient(blockX, blockY)
        grad2 = self._get_gradient(blockX + 1, blockY)
        grad3 = self._get_gradient(blockX + 1, blockY + 1)
        grad4 = self._get_gradient(blockX, blockY + 1)

        dot1 = self._dot((x, y), grad1)
        dot2 = self._dot((x - 1, y), grad2)
        dot3 = self._dot((x - 1, y - 1), grad3)
        dot4 = self._dot((x, y - 1), grad4)

        # Interpolate the results, add 1 and divide by 2 to adjust range
        thresh1 = 1  # 0 - 2
        thresh2 = 2  # > 2
        return (self._lerp(self._lerp(dot3, dot4, self._f(x)), self._lerp(dot2, dot1, self._f(x)), self._f(y)) + thresh1) / thresh2
    
    def create_image(self, width=500, height=500, out_path='./noise.png'):
        '''Exports the perlin noise as a greyscale image'''
        img = Image.new('L', (height, width), 255)
        data = img.load()
        for x in range(height):
            for y in range(width):
                value = self.perlin_noise(x, y)
                data[x, y] = math.floor(value * 255)
        img.save(out_path)

def main():
    freq = 200  # Freq, smaller is more dense
    noise = Perlin(freq, seed='gordie')
    noise.create_image(width=600, height=600, out_path='./noise.png')

if __name__ == '__main__':
    main()