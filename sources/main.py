import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QSizePolicy, QApplication, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from numba import jit, prange
import sys
from PIL import Image as im
from PIL import ImageQt


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def mandel(x, y, max_iter):
    i = 0
    c = complex(x, y)
    z = 0.0j
    for i in prange(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 255


@jit(nopython=True, cache=True, fastmath=True, nogil=True, parallel=True)
def create_fractal(min_x, max_x, min_y, max_y, image, max_iter):
    height, width = image.shape
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in prange(width):
        real = min_x + x * pixel_size_x
        for y in prange(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, max_iter)
            image[y, x] = color


class Renderer:
    def __init__(self, height, width, max_iter):
        self.height = height
        self.width = width
        self.image = None
        self.renderFractal((-2.0, 1.0), (-1.0, 1.0))
        self.max_iter = max_iter

    def getImage(self):
        return self.image

    def renderFractal(self, x: tuple[int, int], y: tuple[int, int]):
        self.image = np.zeros((self.height, self.width), dtype=np.uint8)
        create_fractal(x[0], x[1], y[0], y[1], self.image, 50)


class MandelApp(QWidget):
    def __init__(self):
        super().__init__()

        # Start Animation
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.movePos)
        self.timer.start()

        # Default position
        self.x = [-2.0, 1.0]
        self.y = [-1.0, 1.0]
        self.pos = [self.x, self.y]

        self.zoomin = False

        # Qt init
        self.label = QLabel()
        self.mandel = Renderer(self.height(), self.width(), 100)
        self.hbox = QHBoxLayout(self)
        self.hbox.addWidget(self.label)
        self.renderFractal()
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setPixmap(QPixmap.fromImage(self.getImage()))
        self.show()

    def renderFractal(self):
        self.mandel.renderFractal(self.x, self.y)

    def getImage(self) -> ImageQt:
        return ImageQt.ImageQt(im.fromarray(self.mandel.getImage()))

    def resizeEvent(self, event):
        self.mandel.height = self.height()
        self.mandel.width = self.width()
        self.renderFractal()
        self.label.setPixmap(QPixmap.fromImage(self.getImage()))

    def movePos(self):
        multi = .995 if self.zoomin else 1.005
        if self.pos[0][0] > -0.50 and self.zoomin:
            self.zoomin = False
        elif self.pos[0][0] < -10 and not self.zoomin:
            self.zoomin = True

        self.pos[0][0] *= multi
        self.pos[0][1] *= multi
        self.pos[1][0] *= multi
        self.pos[1][1] *= multi

        self.renderFractal()
        self.label.setPixmap(QPixmap.fromImage(self.getImage()))


app = QApplication(sys.argv)
win = MandelApp()
sys.exit(app.exec_())
