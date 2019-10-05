import tkinter as tk
import os
import sys

r = tk.Tk()


def run():
    os.system('python main.py')


button = tk.Button(r, text='START', width=25, command=run)
button.pack()
r.mainloop()
