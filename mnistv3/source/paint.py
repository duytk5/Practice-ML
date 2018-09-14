from tkinter import *
import io
from PIL import Image

from source import build as mB

class Paint(object):

    DEFAULT_PEN_SIZE = 30.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.solve_button = Button(self.root, text='solve', command=self.use_solve)
        self.solve_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.choose_size_button = Scale(self.root, from_=30, to=30, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=400, height=400)
        self.c.grid(row=1, columnspan=4)

        self.lb = Label(self.root, bg='white', width=25, height= 5 , font=("Courier", 20))
        self.lb.grid(row=2, columnspan=4)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_solve(self):
        self.lb.config(text='solving ... ' )
        self.c.update()
        ps = self.c.postscript(colormode='mono')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        size = 28,28
        img.thumbnail(size, Image.ANTIALIAS)
        # print(img)
        img.save('./input.jpg')
        val, acc = mB.test()
        accstr = ''
        for ii in range(len(acc)):
            ss = '__%d : %.2f\n' % ( ii,  acc[ii]*100)
            accstr  = accstr + ss
        print(accstr)
        self.lb.config(text= 'OUTPUT: '+ str(val))

    def use_eraser(self):
        self.c.create_rectangle(0,0,400,400,fill = 'white')
        #self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color

        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=FALSE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()