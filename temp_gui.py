from tkinter import *
from PIL import Image

from perlin_noise import Perlin

class MainWindow():

    def __init__(self, main):
        main.title('Procedural Map Generator')
        main.resizable(False, False)

        self.image_path = 'noise.png'
        self.image = PhotoImage(file=self.image_path)
        im = Image.open(self.image_path)
        self.width, self.height = im.size

        self.canvas = Canvas(main, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)

        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.image)

        self.button = Button(main, text='Generate', command=self.on_button)
        self.button.grid(row=1, column=0)

        self.slider_01_label = Label(main, text='Frequency: ')
        self.slider_01_label.grid(row=2, column=0, sticky=W)
        self.slider_01 = Scale(main, from_=0, to=150, orient=HORIZONTAL, length=self.width - 150)
        self.slider_01.grid(row=2, column=0)
        

    def on_button(self):
        frequency = self.slider_01.get()
        frequency = 150 - frequency  # Reversing frequency to make more sense
        noise = Perlin(frequency)
        img = noise.create_image(width=600, height=600)
        img_gauss = noise.gaussian_kernel(img)
        img_gauss.save('noise.png')

        self.image = PhotoImage(file=self.image_path)
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)

        
root = Tk()

MainWindow(root)
root.mainloop()