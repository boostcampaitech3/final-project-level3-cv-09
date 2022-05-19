# source: https://infinitt.tistory.com/57

import sys, os, random
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
import random
class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        
        self.axes = fig.add_subplot(211, xlim=(0, 50), ylim=(0, 1024))
        self.axes2 = fig.add_subplot(212, xlim=(0, 50), ylim=(0, 600))

        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
    def compute_initial_figure(self):
        pass
class AnimationWidget(QWidget):
    def __init__(self):
        QMainWindow.__init__(self)
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(self, width=10, height=8, dpi=100)
        vbox.addWidget(self.canvas)
        hbox = QHBoxLayout()
        self.start_button = QPushButton("start", self)
        self.stop_button = QPushButton("stop", self)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        vbox.addLayout(hbox)
        self.setLayout(vbox)


       


        self.x = np.arange(50)
        self.y = np.ones(50, dtype=np.float)*np.nan
        self.line, = self.canvas.axes.plot(self.x, self.y, animated=True, lw=2)

        self.x2 = np.arange(50)
        self.y2 = np.ones(50, dtype=np.float)*np.nan
        self.line2, = self.canvas.axes2.plot(self.x2, self.y2, animated=True,color='red', lw=2)


    def update_line(self, i):
        y = random.randint(0,1024)
        old_y = self.line.get_ydata()
        new_y = np.r_[old_y[1:], y]
        self.line.set_ydata(new_y)
        
        # self.line.set_ydata(y)
        print(self.y)
        return [self.line]

    def update_line2(self, i):
        y2 = random.randint(0,510)
        old_y2 = self.line2.get_ydata()
        new_y2 = np.r_[old_y2[1:], y2]
        self.line2.set_ydata(new_y2)
        return [self.line2]
        # self.line.set_ydata(y)

        
    def on_start(self):
        self.ani = animation.FuncAnimation(self.canvas.figure, self.update_line,blit=True, interval=25)
        self.ani2 = animation.FuncAnimation(self.canvas.figure, self.update_line2,blit=True, interval=25)
    
    def on_stop(self):
        self.ani._stop()
        self.ani2._stop()

if __name__ == "__main__":
    qApp = QApplication(sys.argv)
    aw = AnimationWidget()
    aw.show()
    sys.exit(qApp.exec_())