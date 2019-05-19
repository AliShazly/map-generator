from tkinter import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from voronoi import *
from perlin_noise import Perlin
import main as m

class MainWindow():

    def __init__(self, main):

        main.title('Procedural Map Generator')
        main.resizable(False, False)

        self.image_path = 'images/voro.png'
        self.image = PhotoImage(file=self.image_path)
        im = Image.open(self.image_path)
        self.width, self.height = im.size

        self.canvas = Canvas(main, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)

        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.image)

        self.button = Button(main, text='Generate', command=self.on_button)
        self.button.grid(row=1, column=0)

        self.button = Button(main, text='Reset Params', command=self.reset)
        self.button.grid(row=2, column=0)

        self.slider_01_label = Label(main, text='Frequency: ')
        self.slider_01_label.grid(row=3, column=0, sticky=W)
        self.slider_01 = Scale(main, from_=5, to=150, orient=HORIZONTAL, length=self.width - 150)
        self.slider_01.set(40)
        self.slider_01.grid(row=3, column=0)

        self.slider_02_label = Label(main, text='Size: ')
        self.slider_02_label.grid(row=4, column=0, sticky=W)
        self.slider_02 = Scale(main, from_=2, to=100, orient=HORIZONTAL, length=self.width - 150)
        self.slider_02.set(30)
        self.slider_02.grid(row=4, column=0)

        self.slider_03_label = Label(main, text='Smoothing: ')
        self.slider_03_label.grid(row=5, column=0, sticky=W)
        self.slider_03 = Scale(main, from_=0, to=5, orient=HORIZONTAL, length=self.width - 150)
        self.slider_03.set(2)
        self.slider_03.grid(row=5, column=0)

        self.slider_04_label = Label(main, text='Sigma: ')
        self.slider_04_label.grid(row=6, column=0, sticky=W)
        self.slider_04 = Scale(main, from_=1, to=7, resolution=0.1, orient=HORIZONTAL, length=self.width - 150)
        self.slider_04.set(3.0)
        self.slider_04.grid(row=6, column=0)
        
    def reset(self):
        self.slider_01.set(40)
        self.slider_02.set(30)
        self.slider_03.set(2)
        self.slider_04.set(3.0)

    def on_button(self):
        frequency = self.slider_01.get()
        # frequency = 150 - frequency  # Reversing frequency to make more sense

        size = self.slider_02.get()

        smoothing = self.slider_03.get()

        sigma = self.slider_04.get()

        m.full_frame()  # Removes frame from voronoi matplotlib
        m.main(freq=frequency, lloyds=smoothing, size=size, sigma=sigma)

        self.image = PhotoImage(file=self.image_path)
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)

        
root = Tk()

MainWindow(root)
root.mainloop()
raise SystemExit