import tkinter as tk
from PIL import Image, ImageTk
import cv2
class GuiUtils():
    def update_textbox(self, textbox, text):

        textbox.delete("1.0", "end")  # if you want to remove the old data
        textbox.insert(tk.END, text)

    def askopenimage(self, initialdir=''):
        from tkinter import filedialog as fd
        filepath = fd.askopenfilename(initialdir=initialdir)
        if ProcUtils().isimpath(filepath):
            return cv2.imread(filepath)
        else:
            print('%s is not an image ...' % filepath)

    def im2tkIm(self, im):
        return ImageTk.PhotoImage(Image.fromarray(im))

    def textbox2text(self, textbox):
        return textbox.get("1.0", "end")

    def imshow(self, im, title='viewer', size=(1080, 720)):         # imcomplete
        window = tk.Tk()
        window.geometry('{}x{}'.format(size[0], size[1]))
        window.title(title)


        viewer = tk.Label(window)
        viewer.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, anchor=tk.NW)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(im, size)))
        viewer.configure(image=img, text='something',compound=tk.TOP, font=("Courier", 18), anchor=tk.NW)
        viewer.image = img

        window.mainloop()

