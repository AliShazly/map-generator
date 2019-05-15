from perlin_noise import Perlin
import numpy
from PIL import Image

# https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1/simple-pattern-examples

# img1,img2,img3,img4,img5 = 0,0,0,0,0

# imlist = [img1, img2, img3, img4, img5]

# for idx,i in enumerate(imlist):
#     i.create_image(width=600, height=600, out_path=f'{idx}.png')

im0 = Perlin(2, seed='nu')
im0.create_image(width=300, height=300, out_path='./im0.png')
img0 = Image.open('./im0.png')

im1 = Perlin(8, seed='nu')
im1.create_image(width=300, height=300, out_path='./im1.png')
img1 = Image.open('./im1.png')

im2 = Perlin(16, seed='nu')
im2.create_image(width=300, height=300, out_path='./im2.png')
img2 = Image.open('./im2.png')

im3 = Perlin(32, seed='nu')
im3.create_image(width=300, height=300, out_path='./im3.png')
img3 = Image.open('./im3.png')

im4 = Perlin(64, seed='nu')
im4.create_image(width=300, height=300, out_path='./im4.png')
img4 = Image.open('./im4.png')

im5 = Perlin(128, seed='nu')
im5.create_image(width=300, height=300, out_path='./im5.png')
img5 = Image.open('./im5.png')



w,h = img1.size
arr=numpy.zeros((h,w),numpy.float)

imlist = [img0, img1, img2, img3, img4, img5]

for im in imlist:
    imarr = numpy.array(im,dtype=numpy.float)
    arr = arr+imarr/len(imlist)

arr = numpy.array(numpy.round(arr),dtype=numpy.uint8)

print(arr.shape)
out=Image.fromarray(arr)
out.save("average.png")
