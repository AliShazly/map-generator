from tkinter import *
from PIL import Image

class MainWindow():

    def __init__(self, main):
        main.title('Procedural Map Generator')
        main.resizable(False, False)

        self.image_path = 'map_reference.png'
        self.image = PhotoImage(file=self.image_path)
        im = Image.open(self.image_path)
        self.width, self.height = im.size

        self.canvas = Canvas(main, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)

        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.image)

        self.button = Button(main, text='Generate', command=self.on_button)
        self.button.grid(row=1, column=0)

        self.slider_01_label = Label(main, text='Param 1: ')
        self.slider_01_label.grid(row=2, column=0, sticky=W)
        self.slider_01 = Scale(main, from_=0, to=100, orient=HORIZONTAL, length=self.width - 100)
        self.slider_01.grid(row=2, column=0)
        

    def on_button(self):
        self.image = PhotoImage(file=self.image_path)
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)

        
root = Tk()

MainWindow(root)
root.mainloop()