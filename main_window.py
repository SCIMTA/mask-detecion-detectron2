"""
In this example, we demonstrate how to create simple camera viewer using Opencv3 and PyQt5

Author: Berrouba.A
Last edited: 21 Feb 2018
"""

# import system module
import sys

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget

from ui_main_window import *
from mask_detect import *


class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.view_cam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.control_timer)

    # view camera
    def view_cam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        copy = image.copy()

        faces = predict_detectron(image)

        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.image_label.setScaledContents(False)

        labels = [self.ui.label_1, self.ui.label_2, self.ui.label_3, self.ui.label_4, self.ui.label_5]
        index = 0
        for [[x, y, w, h], c] in faces:
            crop_img = copy[y - 10:h + 10, x - 10:w + 10].copy()
            hh, ww, cc = crop_img.shape
            ss = cc * ww
            # create QImage from image
            qCrop = QImage(crop_img.data, ww, hh, ss, QImage.Format_RGB888)
            labels[index].setPixmap(QPixmap.fromImage(qCrop))
            labels[index].setScaledContents(True)
            index += 1
            if index == 4:
                break

    # show zoom cam
    def zoom_cam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)

        labels = [self.label, self.label_2, self.label_3, self.label_4, self.label_5]
        index = 0

        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.image_label.setScaledContents(True)

    # start/stop timer
    def control_timer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
