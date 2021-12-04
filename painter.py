"""
@author: The Absolute Tinkerer
"""

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QPixmap, QColor
from PyQt5.QtCore import Qt


# Execute this code upon import
app = QApplication(sys.argv)


class Painter(QPainter):
    def __init__(self, width, height, bg_color=QColor(255, 255, 255, 255)):
        """
        Constructor
        """
        super(Painter, self).__init__()

        # Create the image upon which we're going to draw
        self.image = QPixmap(width, height)

        self.image.fill(Qt.transparent)

        # Begin the drawing on the image
        self.begin(self.image)

        self.fillRect(0, 0, width, height, bg_color)

    def saveImage(self, fileName, fmt=None, quality=-1):
        return self.image.save(fileName, fmt, quality)

    def endProgram(self, exit_code=0):
        self.end()
        if exit_code == 0:
            sys.exit(0)
