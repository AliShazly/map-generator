from perlin_noise import Perlin
import numpy
from PIL import Image

# def main():
#     freq = 200  # Freq, smaller is more dense
#     noise = Perlin(freq)
#     noise.create_image(width=600, height=600, out_path='./noise.png')

# im1 = Perlin(25)
# im1.create_image(width=600, height=600, out_path='./im1.png')
img1 = Image.open('./im1.png')

# im2 = Perlin(50)
# im2.create_image(width=600, height=600, out_path='./im2.png')
img2 = Image.open('./im2.png')

# im3 = Perlin(75)
# im3.create_image(width=600, height=600, out_path='./im3.png')
img3 = Image.open('./im3.png')

# im4 = Perlin(100)
# im4.create_image(width=600, height=600, out_path='./im4.png')
img4 = Image.open('./im4.png')

# im5 = Perlin(125)
# im5.create_image(width=600, height=600, out_path='./im5.png')
img5 = Image.open('./im5.png')

w,h = img1.size
arr=numpy.zeros((h,w),numpy.float)

imlist = [img1, img2, img3, img4, img5]

for im in imlist:
    imarr = numpy.array(im,dtype=numpy.float)
    arr = arr+imarr/len(imlist)

arr = numpy.array(numpy.round(arr),dtype=numpy.uint8)

print(arr.shape)
out=Image.fromarray(arr)
out.save("Average.png")
