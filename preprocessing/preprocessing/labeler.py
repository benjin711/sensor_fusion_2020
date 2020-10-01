#!/usr/bin/env python

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QDialog, QGridLayout, QGroupBox,
                             QPushButton, QSlider, QSpinBox, QWidget)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QColor, QPen
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from enum import Enum
import cv2 as cv
import numpy as np
import os
from copy import deepcopy
import json

from utils.utils import load_camera_calib


class Transform(Enum):
    XTRANS = 0
    YTRANS = 1
    ZTRANS = 2
    XROT = 3
    YROT = 4
    ZROT = 5


SuperviselyImageTemplate = {"description": "",
                        "name": "",
                            "size": {
                                "width": 0,
                                "height": 0
                            },
                            "objects": []
                        }
SuperviselyObjectTemplate = {"description": "",
                             "geometryType": "rectangle",
                             "labelerLogin": "ctyfang",
                             "tags": [],
                             "classTitle": "blue_cone",
                             "points": {
                                "exterior": [[0, 0], [1, 1]],
                                "interior": []}
                             }

class QImageViewer(QMainWindow):
    def __init__(self, parent=None):
        super(QImageViewer, self).__init__(parent)
        self.parent = parent

        self.printer = QPrinter()
        self.scaleFactor = 1.0

        self.cone_array = np.zeros((1, 4))              # Raw cone data [color, x, y, z]
        self.cone_kdtree = cKDTree([[0]])                     # Current KDTree of projected cone points with transformation applied
        self.transformationMatrix = np.eye(4)           # Current transformation matrix
        self.camera_matrix = np.eye(3)                  # Current matrix for camera
        self.distortion_coefficients = np.zeros((1, 8))

        self.pnpActive = False
        self.selectionState = "2D"
        self.current2DSelection = [0, 0]
        self.current3DSelection = [0, 0, 0]
        self.boxCoordinates = []
        self.boxDepths = []
        self.coneHeight = 0.325
        self.coneWidth = 0.228

        self.painter = QPainter()
        self.image = None
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.mousePressEvent = self.getPixel
        self.imageLabel.wheelEvent = self.scrollingZoom

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(800, 600)

    def scrollingZoom(self, event):
        degrees = event.angleDelta() / 8
        if degrees.y() > 0:
            self.zoomIn()
        elif degrees.y() < 0:
            self.zoomOut()

    def updateConeKDTree(self):
        """Takes the raw cone XYZ, transform using current transformation
        matrix and project onto the image. Store pixel coordinates in tree."""
        cone_xyz = self.cone_array[:, 1:]
        cone_ones = np.ones((cone_xyz.shape[0], 1))
        homo_cone_xyz = np.concatenate([cone_xyz, cone_ones], axis=1)
        cone_xyz_new = np.transpose(np.matmul(self.transformationMatrix, np.transpose(homo_cone_xyz)))[:, :3]
        homo_cone_pixels = np.transpose(np.matmul(self.camera_matrix, np.transpose(cone_xyz_new)))
        homo_cone_pixels /= homo_cone_pixels[:, 2].reshape((-1, 1))
        cone_pixels = homo_cone_pixels[:, :2]
        self.cone_kdtree = cKDTree(cone_pixels)

    def updatePreviewWithTransform(self, transformationMatrix):
        """Apply transformationMatrix to cone XYZ, draw the points"""
        if self.image:
            image_cv = self.convertQImageToOpenCV(self.image)
            self.transformationMatrix = transformationMatrix

            # Generate new kdtree for the cones
            self.updateConeKDTree()

            self.imageWithCones = self.convertOpenCVToQImage(self.drawCones(image_cv))
            self.imageLabel.setPixmap(QPixmap.fromImage(self.imageWithCones))
            self.scaleImage(self.scaleFactor, adjustScrollbar=False)
            # self.resize(self.imageWithCones.width() * self.scaleFactor,
            #             self.imageWithCones.height() * self.scaleFactor)
            self.imageLabel.resize(self.imageWithCones.width() * self.scaleFactor,
                                   self.imageWithCones.height() * self.scaleFactor)

    def computeBoxDimensions(self, point_3D):
        distance = np.linalg.norm(np.asarray(point_3D), ord=2)
        focalLength_px = np.mean([self.camera_matrix[0, 0], self.camera_matrix[1, 1]])
        height_px = focalLength_px*self.coneHeight/distance
        width_px = focalLength_px*self.coneWidth/distance
        return [height_px, width_px, distance]

    def drawBoxes(self, image):
        """Draw boxes on an opencv formatted image.
        Return the image with boxes drawn on it."""
        new_image = deepcopy(image)
        cone_color, cone_xyz = self.cone_array[:, 0], self.cone_array[:, 1:]
        ones = np.ones((cone_color.shape[0], 1))
        cone_xyz_homo = np.concatenate([cone_xyz, ones], axis=1)
        cone_xyz_transformed = np.transpose(
            np.matmul(self.transformationMatrix, np.transpose(cone_xyz_homo)))
        cone_points = cone_xyz_transformed[:, :3]

        box_coordinates = []
        box_depths = []
        for pixel_idx in range(self.cone_kdtree.data.shape[0]):
            tip_point = cone_points[pixel_idx, :]
            tip_pixel = self.cone_kdtree.data[pixel_idx, :]
            height_px, width_px, depth_m = self.computeBoxDimensions(tip_point)
            top_left_pixel = (round(tip_pixel[0]-width_px//2), round(tip_pixel[1]))
            bot_right_pixel = (round(tip_pixel[0]+width_px//2), round(tip_pixel[1]+height_px))
            box_coordinates.append([top_left_pixel, bot_right_pixel])
            box_depths.append(depth_m)
            cv.rectangle(new_image, top_left_pixel, bot_right_pixel, (0, 255, 0), 2)

        self.boxCoordinates = box_coordinates
        self.boxDepths = box_depths
        return new_image

    def drawCones(self, image):
        """Draw cone points on an opencv formatted image.
        Return the image with points drawn on it"""
        new_image = deepcopy(image)
        cone_color, cone_xyz = self.cone_array[:, 0], self.cone_array[:, 1:]
        ones = np.ones((cone_xyz.shape[0], 1))
        cone_xyz_homo = np.concatenate([cone_xyz, ones], axis=1)
        cone_xyz_transformed = np.transpose(np.matmul(self.transformationMatrix, np.transpose(cone_xyz_homo)))
        cone_pixels = np.transpose(np.matmul(self.camera_matrix, np.transpose(cone_xyz_transformed[:, :3])))
        cone_pixels /= cone_pixels[:, 2].reshape((-1, 1))

        for pixel_idx in range(cone_pixels.shape[0]):
            pixel = cone_pixels[pixel_idx, :]
            cv.circle(new_image, (int(pixel[0]), int(pixel[1])), 2, (255, 0, 0), -1)

        return new_image

    def showBoxes(self):
        """For the current projected cone points, generate an image with both
            projected cone points and boxes."""

        if self.image:
            image_cv = self.convertQImageToOpenCV(self.image)
            image_cones_cv = self.drawCones(image_cv)
            image_boxes_cv = self.drawBoxes(image_cones_cv)
            self.imageWithBoxes = self.convertOpenCVToQImage(image_boxes_cv)

            # Update imageLabel widget
            self.imageLabel.setPixmap(QPixmap.fromImage(self.imageWithBoxes))
            self.scaleImage(self.scaleFactor, adjustScrollbar=False)
            # self.resize(self.imageWithCones.width() * self.scaleFactor,
            #             self.imageWithCones.height() * self.scaleFactor)
            self.imageLabel.resize(self.imageWithCones.width() * self.scaleFactor,
                                   self.imageWithCones.height() * self.scaleFactor)
        else:
            print("Image need to be loaded.")

    def updateImageAndCones(self, imagePath, conePath):
        """Load the raw image and cones. Undistort the image and store it
           as a QImage. Assumes that the camera matrix and distortion
           coefficients have been assigned."""

        if imagePath and conePath:
            # Save undistorted image
            raw_image = QImage(imagePath)
            image_cv = self.convertQImageToOpenCV(raw_image)
            image_cv = cv.undistort(image_cv, self.camera_matrix, self.distortion_coefficients)
            self.image = self.convertOpenCVToQImage(image_cv)

            # Save cones and generate kdTree
            self.cone_array = np.fromfile(conePath).reshape((-1, 4))
            self.updateConeKDTree()

            # Save image with cones
            image_cones_cv = self.drawCones(image_cv)
            self.imageWithCones = self.convertOpenCVToQImage(image_cones_cv)

            # Update imageLabel widget
            self.imageLabel.setPixmap(QPixmap.fromImage(self.imageWithCones))
            self.scaleFactor = 1.0
            self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()
            self.resize(self.image.width() * self.scaleFactor,
                        self.image.height() * self.scaleFactor)
            self.imageLabel.resize(self.image.width() * self.scaleFactor,
                                   self.image.height() * self.scaleFactor)

    def convertQImageToOpenCV(self, image_qt):
        new_image = image_qt.convertToFormat(4)
        width = new_image.width()
        height = new_image.height()
        ptr = new_image.bits()
        ptr.setsize(new_image.byteCount())
        image_cv = np.array(ptr).reshape((height, width, 4))
        return image_cv

    def convertOpenCVToQImage(self, image_cv):
        return QImage(image_cv.data, image_cv.shape[1], image_cv.shape[0], QImage.Format_ARGB32)

    def getPixel(self, event):
        x = event.pos().x() # For the scaled image
        y = event.pos().y()
        x = round(x/self.scaleFactor)
        y = round(y/self.scaleFactor)

        # Update image to show selected pixel OR nearest projected pixel
        image_cv = self.convertQImageToOpenCV(self.imageWithCones)  # Get raw image, convert to CV format
        if self.selectionState == "3D":
            _, neighborIndex = self.cone_kdtree.query([x, y], 1)
            self.current3DSelection = self.cone_array[neighborIndex, 1:].tolist()
            x, y = self.cone_kdtree.data[neighborIndex, :]
            x = round(x)
            y = round(y)
        else:
            self.current2DSelection = [x, y]
        cv.circle(image_cv, (x, y), 3, (255, 0, 0), -1)
        new_image = self.convertOpenCVToQImage(image_cv)        # Convert back to QImage
        self.imageLabel.setPixmap(QPixmap.fromImage(new_image)) # Show it
        self.scaleImage(self.scaleFactor, adjustScrollbar=False)

    def set_datafolder(self):
        dirName = QFileDialog.getExistingDirectory(self,
                                                   'QFileDialog.getExistingDirectory()',
                                                   '',)
        self.dirName = dirName

    def open_calibration(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  'QFileDialog.getOpenFileName()',
                                                  '',
                                                  'YAML (*.yaml)',
                                                  options=options)

        if fileName:
            calibration_data = load_camera_calib(fileName)
            self.camera_matrix = calibration_data['camera_matrix']

    def open_cones(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'bin (*.bin)', options=options)
        if fileName:
            self.cone_array = np.fromfile(fileName).reshape((-1, 4))

    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        if fileName:
            image = QImage(fileName)
            self.image = image
            self.imageWithCones = image

            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            self.resize(image.width()*self.scaleFactor, image.height()*self.scaleFactor)
            self.imageLabel.resize(image.width()*self.scaleFactor, image.height()*self.scaleFactor)

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleFactor *= 1.25
        self.imageLabel.adjustSize()
        self.scaleImage(self.scaleFactor)

    def zoomOut(self):
        self.scaleFactor *= 0.8
        self.imageLabel.adjustSize()
        self.scaleImage(self.scaleFactor)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleImage(1/self.scaleFactor)

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")

    def createActions(self):
        self.useLastPnPAct = QAction("&Apply last solvePnP Solution...", self.parent, shortcut="Ctrl+P", enabled=True, triggered=self.parent.lastSolvePnPSolutionCB)
        self.generateLabelAct = QAction("&Generate labels...", self.parent, shortcut="Ctrl+L", enabled=True, triggered=self.generateLabels)
        self.showBoxesAct = QAction("&Show boxes...", self.parent, shortcut="Ctrl+B", enabled=True, triggered=self.showBoxes)
        self.confirmSelectionAct = QAction("&Confirm selected point and append to correspondences...", self.parent, shortcut="Ctrl+Space", enabled=True, triggered=self.parent.updateCorrespondencesCB)
        self.nextImageAct = QAction("&Go to the next image...", self.parent, shortcut=Qt.Key_Right, enabled=True, triggered=self.parent.nextImageCB)
        self.prevImageAct = QAction("&Go to the prev image...", self.parent, shortcut=Qt.Key_Left, enabled=True, triggered=self.parent.prevImageCB)
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.openConesAct = QAction("&Load Cones...", self, shortcut="Ctrl+C", triggered=self.open_cones)
        self.openCalibAct = QAction("&Load Camera Calibration...", self, shortcut="Ctrl+C", triggered=self.open_calibration)
        self.setDirectoryAct = QAction("&Set data folder...", self, shortcut="Ctr+D", triggered=self.set_datafolder)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut=Qt.Key_Up, enabled=True, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut=Qt.Key_Down, enabled=True, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.useLastPnPAct)
        self.fileMenu.addAction(self.generateLabelAct)
        self.fileMenu.addAction(self.showBoxesAct)
        self.fileMenu.addAction(self.confirmSelectionAct)
        self.fileMenu.addAction(self.nextImageAct)
        self.fileMenu.addAction(self.prevImageAct)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        self.fileMenu.addAction(self.openConesAct)
        self.fileMenu.addAction(self.openCalibAct)
        self.fileMenu.addAction(self.setDirectoryAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        # self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        # self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        # self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor, adjustScrollbar=True):
        self.imageLabel.resize(factor * self.imageLabel.pixmap().size())

        if adjustScrollbar:
            self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
            self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

    def generateLabels(self):
        """Convert the box coordinates and depths to a Supervisely formatted
        JSON file."""
        newImageLabel = deepcopy(SuperviselyImageTemplate)
        newImageLabel["description"] = str(self.parent.imageIndex).zfill(8)+'.png'
        newImageLabel["size"]["width"] = 640
        newImageLabel["size"]["height"] = 2592

        boxDepths = np.asarray(self.boxDepths).reshape((-1, 1))
        boxClasses = self.cone_array[:, 0]
        boxCoordinates = np.asarray(self.boxCoordinates).reshape((-1, 2, 2))
        for boxIndex in range(boxDepths.shape[0]):
            newBoxLabel = deepcopy(SuperviselyObjectTemplate)

            if boxClasses[boxIndex] == 0:
                newBoxLabel["classTitle"] = "blue_cone"
            elif boxClasses[boxIndex] == 1:
                newBoxLabel["classTitle"] = "yellow_cone"
            else:
                continue
            newBoxLabel["points"]["exterior"][0] = boxCoordinates[boxIndex, 0, :].tolist()
            newBoxLabel["points"]["exterior"][1] = boxCoordinates[boxIndex, 1, :].tolist()
            newBoxLabel["description"] = str(boxDepths[boxIndex])
            newImageLabel["objects"].append(newBoxLabel)

        if os.path.exists(self.parent.labelPath):
            print("[ERROR] Tried to overwrite label file")
        else:
            with open(self.parent.labelPath, 'w') as fp:
                json.dump(newImageLabel, fp, indent=4)

class LabelerControls(QDialog):
    def __init__(self, parent=None):
        super(LabelerControls, self).__init__(parent)

        self.cameraName = "forward"
        self.imageDir = ""
        self.coneDir = ""
        self.imageIndex = 0
        self.imagePreview = QImageViewer(parent=self)
        self.originalPalette = QApplication.palette()

        # Initial Transform
        self.eulerXYZ = [0., 0., 0.]
        self.transformationMatrix = np.eye(4)
        self.lastSolvePnPSolution = np.eye(4)

        # Tuning limits (meters, degrees) and number of ticks to use
        self.translation_range = 1.5
        self.translation_res = 500
        self.rotation_range = 45.
        self.rotation_res = 1000

        # Correspondences for solvePnP
        self.correspondences2D = []
        self.correspondences3D = []

        self.createInitializationGroupBox()
        self.createTuningGroupBox()
        self.createSolvePnPGroupBox()
        self.createLabelGroupBox()
        self.createArrowKeysGroupBox()

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.initializationGroupBox, 0, 0, 1, 1)
        mainLayout.addWidget(self.arrowKeysGroupBox, 1, 0, 1, 1)
        mainLayout.addWidget(self.tuningGroupBox, 0, 1, 4, 1)
        mainLayout.addWidget(self.solvePnPGroupBox, 2, 0, 1, 1)
        mainLayout.addWidget(self.labelGroupBox, 3, 0, 1, 1)
        self.setLayout(mainLayout)
        self.setWindowTitle("Labeler Settings")
        self.setFixedSize(800, 600)

    def updateCorrespondencesCB(self):
        """Pull correspondence from imagePreview. Update labels."""
        self.confirmPushButton.click()

    def nextImageCB(self):
        # TODO: Check the number of images available
        self.imageIndex += 1
        if self.imageIndex > self.maxImageIndex:
            print("Already at the maximum index.")
            self.imageIndex -= 1
        else:
            self.updatePreviewFromIndexCB(self.imageIndex)

    def prevImageCB(self):
        self.imageIndex -= 1
        if self.imageIndex < 0:
            print("Already at the beginning.")
            self.imageIndex = 0
        else:
            self.updatePreviewFromIndexCB(self.imageIndex)

    def updatePreviewFromIndexCB(self, newIndex):
        self.imageIndex = newIndex
        self.imagePath = os.path.join(self.imageDir, str(self.imageIndex).zfill(8) + '.png')
        self.conePath = os.path.join(self.coneDir, str(self.imageIndex).zfill(8) + '.bin')
        self.labelPath = os.path.join(self.labelDir, str(self.imageIndex).zfill(8)+'.png.json')

        self.imagePreview.updateImageAndCones(self.imagePath, self.conePath)
        self.imagePreview.updatePreviewWithTransform(self.transformationMatrix)

    def selectImageDirectoryCB(self, textbox):
        dirName = QFileDialog.getExistingDirectory(self,
                                                   'QFileDialog.getExistingDirectory()',
                                                   '', )
        if dirName:
            self.imageDir = dirName
            print('Image directory changed to:' + self.imageDir)
            textbox.setText(self.imageDir)
            # Check the number of images available
            self.maxImageIndex = len(os.listdir(self.imageDir))-1

        else:
            print("Invalid directory specified.")
            textbox.setText("Invalid directory")

    def selectConeDirectoryCB(self, textbox):
        dirName = QFileDialog.getExistingDirectory(self,
                                                   'QFileDialog.getExistingDirectory()',
                                                   '', )
        if dirName:
            self.coneDir = dirName
            print('Cone directory changed to:' + self.coneDir)
            textbox.setText(self.coneDir)

        else:
            print("Invalid directory specified.")
            textbox.setText("Invalid directory")

    def selectBaseDirectoryCB(self, textbox):
        dirName = QFileDialog.getExistingDirectory(self,
                                                   'QFileDialog.getExistingDirectory()',
                                                   '', )
        if dirName:
            self.baseDir = dirName
            print('Base directory changed to:' + self.baseDir)

            head, pathEnd = os.path.split(self.baseDir)
            head, pathMid = os.path.split(head)
            _, pathTop = os.path.split(head)
            textbox.setText(os.path.join(pathTop, pathMid, pathEnd))

            # Update the other paths
            self.imageDir = os.path.join(self.baseDir,
                                        self.cameraName + "_camera_filtered")
            self.maxImageIndex = len(os.listdir(self.imageDir)) - 1
            self.calibPath = os.path.join('resources',
                                          self.cameraName + '.yaml')
            self.calibration = load_camera_calib(self.calibPath)
            self.imagePreview.camera_matrix = self.calibration["camera_matrix"]
            self.imagePreview.distortion_coefficients = self.calibration[
                "distortion_coefficients"]
            self.coneDir = os.path.join(self.baseDir,
                                        self.cameraName + "_cones_filtered")

            self.labelDir = os.path.join(self.baseDir,
                                         self.cameraName + "_labels")
            if not os.path.exists(self.labelDir):
                os.makedirs(self.labelDir)

        else:
            print("Invalid directory specified.")
            textbox.setText("Invalid directory")

    def selectCameraCB(self, cameraComboBox):
        """Update the camera name. Use it to update the calibration path, image directory, and cone directory."""
        self.cameraName = cameraComboBox.currentText()
        self.imageDir = os.path.join(self.baseDir, self.cameraName + "_camera_filtered")
        self.calibPath = os.path.join('resources', self.cameraName + '.yaml')
        self.calibration = load_camera_calib(self.calibPath)
        print(self.calibration)
        self.coneDir = os.path.join(self.baseDir, self.cameraName + "_cones_filtered")
        self.imagePreview.camera_matrix = self.calibration["camera_matrix"]
        self.imagePreview.distortion_coefficients = self.calibration["distortion_coefficients"]

    def createInitializationGroupBox(self):
        self.initializationGroupBox = QGroupBox("Image Selection")

        baseDirSelectLabel = QLabel(self.initializationGroupBox)
        baseDirSelectLabel.setText("Base Directory")
        baseDirSelectTextbox = QLabel(self.initializationGroupBox)
        baseDirSelectTextbox.setText("Please select base directory")
        baseDirSelectButton = QPushButton(self.initializationGroupBox)
        baseDirSelectButton.setDefault(True)
        baseDirSelectButton.setText("Select Base Dir")
        baseDirSelectButton.clicked.connect(lambda: self.selectBaseDirectoryCB(baseDirSelectTextbox))

        cameraComboLabel = QLabel(self.initializationGroupBox)
        cameraComboLabel.setText("Camera Selection")
        cameraComboBox = QComboBox(self.initializationGroupBox)
        cameraComboBox.addItems(["forward", "left", "right"])
        cameraComboBox.currentIndexChanged.connect(lambda: self.selectCameraCB(cameraComboBox))

        spinBoxLabel = QLabel(self.initializationGroupBox)
        spinBoxLabel.setText("Frame Index")
        spinBox = QSpinBox(self.initializationGroupBox)
        spinBox.setValue(0)

        changeImageButton = QPushButton("Update Labeler Configuration and Image")
        changeImageButton.setDefault(True)
        changeImageButton.clicked.connect(
            lambda: self.updatePreviewFromIndexCB(spinBox.value()))

        layout = QGridLayout()
        layout.addWidget(baseDirSelectLabel, 1, 0, 1, 1)
        layout.addWidget(baseDirSelectTextbox, 1, 1, 1, 1)
        layout.addWidget(baseDirSelectButton, 2, 0, 1, 2)

        layout.addWidget(cameraComboLabel, 3, 0, 1, 1)
        layout.addWidget(cameraComboBox, 3, 1, 1, 1)

        layout.addWidget(spinBoxLabel, 5, 0, 1, 1)
        layout.addWidget(spinBox, 5, 1, 1, 1)
        layout.addWidget(changeImageButton, 6, 0, 1, 2)
        self.initializationGroupBox.setLayout(layout)

    def updateRotation(self, transformName, newValue):
        if transformName == Transform.XROT:
            self.eulerXYZ[0] = newValue
        elif transformName == Transform.YROT:
            self.eulerXYZ[1] = newValue
        elif transformName == Transform.ZROT:
            self.eulerXYZ[2] = newValue
        newRotation = Rotation.from_euler('xyz', self.eulerXYZ, degrees=True).as_matrix()
        self.transformationMatrix[:3, :3] = newRotation

    def updateTransform(self, transformName, tickValue, textBox):
        if transformName in [Transform.XTRANS, Transform.YTRANS, Transform.ZTRANS]:
            newValue = (tickValue+1 - (self.translation_res)/2)/((self.translation_res)/2)*self.translation_range

            if transformName == Transform.XTRANS:
                self.transformationMatrix[0, 3] = newValue
            elif transformName == Transform.YTRANS:
                self.transformationMatrix[1, 3] = newValue
            else:
                self.transformationMatrix[2, 3] = newValue

            self.imagePreview.updatePreviewWithTransform(self.transformationMatrix)
            textBox.setText(str(f"{newValue:.2f}"))

        elif transformName in [Transform.XROT, Transform.YROT, Transform.ZROT]:
            newValue = (tickValue + 1 - (self.rotation_res) / 2) / (
                        (self.rotation_res) / 2) * self.rotation_range
            self.updateRotation(transformName, newValue)
            self.imagePreview.updatePreviewWithTransform(self.transformationMatrix)
            textBox.setText(str(f"{newValue:.2f}"))
        else:
            print("Invalid transformation specified")
            return

    def resetTuning(self):
        self.sliderXTrans.setValue(self.translation_res // 2 - 1)
        self.sliderYTrans.setValue(self.translation_res // 2 - 1)
        self.sliderZTrans.setValue(self.translation_res // 2 - 1)
        self.sliderXRot.setValue(self.rotation_res // 2 - 1)
        self.sliderYRot.setValue(self.rotation_res // 2 - 1)
        self.sliderZRot.setValue(self.rotation_res // 2 - 1)

        self.updateTransform(Transform.XTRANS, self.sliderXTrans.value(),
                             self.sliderXTransLabel)
        self.updateTransform(Transform.YTRANS, self.sliderYTrans.value(),
                             self.sliderYTransLabel)
        self.updateTransform(Transform.ZTRANS, self.sliderZTrans.value(),
                             self.sliderZTransLabel)
        self.updateTransform(Transform.XROT, self.sliderXRot.value(),
                             self.sliderXRotLabel)
        self.updateTransform(Transform.YROT, self.sliderYRot.value(),
                             self.sliderYRotLabel)
        self.updateTransform(Transform.ZROT, self.sliderZRot.value(),
                             self.sliderZRotLabel)

    def lockinTransformCB(self):
        """Apply the transform to the cone xyz, reset the transform to identity"""
        cone_xyz = self.imagePreview.cone_array[:, 1:]
        cone_ones = np.ones((cone_xyz.shape[0], 1))
        homo_cone_xyz = np.concatenate([cone_xyz, cone_ones], axis=1)
        homo_cone_xyz_new = np.transpose(np.matmul(self.imagePreview.transformationMatrix, np.transpose(homo_cone_xyz)))
        cone_xyz_new = homo_cone_xyz_new[:, :3]
        self.imagePreview.cone_array[:, 1:] = cone_xyz_new
        self.resetTuning()
        self.transformationMatrix = np.eye(4)
        self.imagePreview.updatePreviewWithTransform(self.transformationMatrix)

    def lastSolvePnPSolutionCB(self):
        self.transformationMatrix = self.lastSolvePnPSolution
        self.imagePreview.updatePreviewWithTransform(self.transformationMatrix)
        self.lockinTransformCB()

    def createTuningGroupBox(self):
        self.tuningGroupBox = QGroupBox("Transform Manual Tuning")
        numSliderRows = 5

        # Create widgets (sliders and labels)
        XTransLabel = QLabel(self.tuningGroupBox)
        XTransLabel.setMargin(0)
        XTransLabel.setText("X-T")
        YTransLabel = QLabel(self.tuningGroupBox)
        YTransLabel.setMargin(0)
        YTransLabel.setText("Y-T")
        ZTransLabel = QLabel(self.tuningGroupBox)
        ZTransLabel.setMargin(0)
        ZTransLabel.setText("Z-T")
        XRotLabel = QLabel(self.tuningGroupBox)
        XRotLabel.setMargin(0)
        XRotLabel.setText("X-R")
        YRotLabel = QLabel(self.tuningGroupBox)
        YRotLabel.setMargin(0)
        YRotLabel.setText("Y-R")
        ZRotLabel = QLabel(self.tuningGroupBox)
        ZRotLabel.setMargin(0)
        ZRotLabel.setText("Z-R")

        self.sliderXTransLabel = QLabel(self.tuningGroupBox)
        self.sliderYTransLabel = QLabel(self.tuningGroupBox)
        self.sliderZTransLabel = QLabel(self.tuningGroupBox)
        self.sliderXRotLabel = QLabel(self.tuningGroupBox)
        self.sliderYRotLabel = QLabel(self.tuningGroupBox)
        self.sliderZRotLabel = QLabel(self.tuningGroupBox)

        self.sliderXTrans = QSlider(Qt.Vertical, self.tuningGroupBox)
        self.sliderXTrans.setMinimum(0)
        self.sliderXTrans.setMaximum(self.translation_res)
        self.sliderXTrans.setValue(self.translation_res//2 - 1)
        self.sliderYTrans = QSlider(Qt.Vertical, self.tuningGroupBox)
        self.sliderYTrans.setMinimum(0)
        self.sliderYTrans.setMaximum(self.translation_res)
        self.sliderYTrans.setValue(self.translation_res // 2 - 1)
        self.sliderZTrans = QSlider(Qt.Vertical, self.tuningGroupBox)
        self.sliderZTrans.setMinimum(0)
        self.sliderZTrans.setMaximum(self.translation_res)
        self.sliderZTrans.setValue(self.translation_res // 2 - 1)

        self.sliderXRot = QSlider(Qt.Vertical, self.tuningGroupBox)
        self.sliderXRot.setMinimum(0)
        self.sliderXRot.setMaximum(self.rotation_res)
        self.sliderXRot.setValue(self.rotation_res // 2 - 1)
        self.sliderXRot.size()
        self.sliderYRot = QSlider(Qt.Vertical, self.tuningGroupBox)
        self.sliderYRot.setMinimum(0)
        self.sliderYRot.setMaximum(self.rotation_res)
        self.sliderYRot.setValue(self.rotation_res // 2 - 1)
        self.sliderZRot = QSlider(Qt.Vertical, self.tuningGroupBox)
        self.sliderZRot.setMinimum(0)
        self.sliderZRot.setMaximum(self.rotation_res)
        self.sliderZRot.setValue(self.rotation_res // 2 - 1)

        # Button for locking in the transform (Applying transform directly to cone data)
        lockinTransformButton = QPushButton(self.tuningGroupBox)
        lockinTransformButton.setText("Lock-in Transform (Ctrl+T)")
        lockinTransformButton.clicked.connect(lambda: self.lockinTransformCB())
        lockinTransformButton.setShortcut("Ctrl+T")

        lastTransformButton = QPushButton(self.tuningGroupBox)
        lastTransformButton.setText("Use Previous solvePnP Solution (Ctrl+P)")
        lastTransformButton.clicked.connect(lambda: self.lastSolvePnPSolutionCB())
        lastTransformButton.setShortcut("Ctrl+P")

        # Callbacks
        self.sliderXTrans.valueChanged.connect(
            lambda: self.updateTransform(Transform.XTRANS,
                                         self.sliderXTrans.value(),
                                         self.sliderXTransLabel))
        self.sliderYTrans.valueChanged.connect(
            lambda: self.updateTransform(Transform.YTRANS,
                                         self.sliderYTrans.value(),
                                         self.sliderYTransLabel))
        self.sliderZTrans.valueChanged.connect(
            lambda: self.updateTransform(Transform.ZTRANS,
                                         self.sliderZTrans.value(),
                                         self.sliderZTransLabel))
        self.sliderXRot.valueChanged.connect(
            lambda: self.updateTransform(Transform.XROT,
                                         self.sliderXRot.value(),
                                         self.sliderXRotLabel))
        self.sliderYRot.valueChanged.connect(
            lambda: self.updateTransform(Transform.YROT,
                                         self.sliderYRot.value(),
                                         self.sliderYRotLabel))
        self.sliderZRot.valueChanged.connect(
            lambda: self.updateTransform(Transform.ZROT,
                                         self.sliderZRot.value(),
                                         self.sliderZRotLabel))
        # Add to layout
        layout = QGridLayout()
        layout.addWidget(XTransLabel, 0, 0, 1, 1)
        layout.addWidget(YTransLabel, 0, 1, 1, 1)
        layout.addWidget(ZTransLabel, 0, 2, 1, 1)
        layout.addWidget(XRotLabel, 0, 3, 1, 1)
        layout.addWidget(YRotLabel, 0, 4, 1, 1)
        layout.addWidget(ZRotLabel, 0, 5, 1, 1)
        layout.addWidget(self.sliderXTransLabel, 1, 0, 1, 1)
        layout.addWidget(self.sliderYTransLabel, 1, 1, 1, 1)
        layout.addWidget(self.sliderZTransLabel, 1, 2, 1, 1)
        layout.addWidget(self.sliderXRotLabel, 1, 3, 1, 1)
        layout.addWidget(self.sliderYRotLabel, 1, 4, 1, 1)
        layout.addWidget(self.sliderZRotLabel, 1, 5, 1, 1)
        layout.addWidget(self.sliderXTrans, 2, 0, numSliderRows, 1)
        layout.addWidget(self.sliderYTrans, 2, 1, numSliderRows, 1)
        layout.addWidget(self.sliderZTrans, 2, 2, numSliderRows, 1)
        layout.addWidget(self.sliderXRot, 2, 3, numSliderRows, 1)
        layout.addWidget(self.sliderYRot, 2, 4, numSliderRows, 1)
        layout.addWidget(self.sliderZRot, 2, 5, numSliderRows, 1)
        layout.addWidget(lockinTransformButton, numSliderRows+2, 0, 1, 6)
        layout.addWidget(lastTransformButton, numSliderRows+3, 0, 1, 6)
        self.tuningGroupBox.setLayout(layout)

        # Init the transforms
        self.updateTransform(Transform.XTRANS, self.sliderXTrans.value(),
                             self.sliderXTransLabel)
        self.updateTransform(Transform.YTRANS, self.sliderYTrans.value(),
                             self.sliderYTransLabel)
        self.updateTransform(Transform.ZTRANS, self.sliderZTrans.value(),
                             self.sliderZTransLabel)
        self.updateTransform(Transform.XROT, self.sliderXRot.value(),
                             self.sliderXRotLabel)
        self.updateTransform(Transform.YROT, self.sliderYRot.value(),
                             self.sliderYRotLabel)
        self.updateTransform(Transform.ZROT, self.sliderZRot.value(),
                             self.sliderZRotLabel)

    def clearCorrespondencesCB(self, modeLabel, counterLabel):
        """Reset the correspondence list. Update the selectionState.
        Update the QLabels."""
        self.correspondences2D = []
        self.correspondences3D = []
        self.selectionState = "2D"
        self.imagePreview.selectionState = self.selectionState
        modeLabel.setText("Being Selected: 2D")
        counterLabel.setText("# Correspondences: 0")

    def confirmCorrespondencesCB(self, modeLabel, counterLabel):
        if self.selectionState == "2D":
            self.correspondences2D.append(self.imagePreview.current2DSelection)
            self.selectionState = "3D"
            self.imagePreview.selectionState = self.selectionState
            modeLabel.setText("Being Selected: 3D")

        else:
            self.correspondences3D.append(self.imagePreview.current3DSelection)
            self.selectionState = "2D"
            self.imagePreview.selectionState = self.selectionState
            modeLabel.setText("Being Selected: 2D")
            counterLabel.setText("# Correspondences: " + str(len(self.correspondences3D)))

    def solvePnPCB(self):
        """Use current correspondences to run solvePnP, update the transformation
        then update the image preview"""
        points = np.asarray(self.correspondences3D).reshape((-1, 1, 3)).astype(np.float32)
        pixels = np.asarray(self.correspondences2D).reshape((-1, 1, 2)).astype(np.float32)
        retval, rvec, tvec, inliers = cv.solvePnPRansac(points, pixels,
                                                        self.calibration["camera_matrix"],
                                                        None,
                                                        iterationsCount=100000000,
                                                        flags=cv.SOLVEPNP_ITERATIVE,
                                                        reprojectionError=4.0,
                                                        confidence=0.999)
        newRotMat, _ = cv.Rodrigues(rvec)
        self.transformationMatrix[:3, :3] = newRotMat
        self.transformationMatrix[:3, 3] = tvec.reshape((3,))
        self.imagePreview.updatePreviewWithTransform(self.transformationMatrix)
        self.lastSolvePnPSolution = deepcopy(self.transformationMatrix)
        self.lockinTransformCB()

    def createSolvePnPGroupBox(self):
        self.solvePnPGroupBox = QGroupBox("SolvePnP")

        self.selectionState = "2D"
        self.imagePreview.selectionState = self.selectionState
        modeLabel = QLabel(self.solvePnPGroupBox)
        modeLabel.setText("Being Selected: 2D")

        counterLabel = QLabel(self.solvePnPGroupBox)
        counterLabel.setText("# Correspondences: 0")

        clearPushButton = QPushButton(self.solvePnPGroupBox)
        clearPushButton.setText("Clear All Selections")
        clearPushButton.clicked.connect(lambda: self.clearCorrespondencesCB(modeLabel, counterLabel))

        self.confirmPushButton = QPushButton(self.solvePnPGroupBox)
        self.confirmPushButton.setText("Confirm Current Selection (Ctrl+Space)")
        self.confirmPushButton.clicked.connect(
            lambda: self.confirmCorrespondencesCB(modeLabel, counterLabel))
        self.confirmPushButton.setShortcut(Qt.Key_Y)
        self.confirmPushButton.setShortcutEnabled(True)

        self.solvePushButton = QPushButton(self.solvePnPGroupBox)
        self.solvePushButton.setText("solvePnP && Apply")
        self.solvePushButton.clicked.connect(
            lambda: self.solvePnPCB())
        self.solvePushButton.setShortcut("S")

        layout = QGridLayout()
        layout.addWidget(modeLabel, 0, 0, 1, 1)
        layout.addWidget(counterLabel, 1, 0, 1, 1)
        layout.addWidget(self.confirmPushButton, 2, 0, 1, 1)
        layout.addWidget(self.solvePushButton, 3, 0, 1, 1)
        layout.addWidget(clearPushButton, 4, 0, 1, 1)
        self.solvePnPGroupBox.setLayout(layout)

    def createLabelGroupBox(self):
        self.labelGroupBox = QGroupBox("Labeling")

        self.showBoxesPushButton = QPushButton(self.labelGroupBox)
        self.showBoxesPushButton.setText("Preview Boxes (Ctrl+B)")
        self.showBoxesPushButton.clicked.connect(lambda: self.imagePreview.showBoxes())
        self.showBoxesPushButton.setShortcut("Ctrl+B")

        self.generateLabelsPushButton = QPushButton(self.labelGroupBox)
        self.generateLabelsPushButton.setText("Build && Save Labels (Ctrl+L)")
        self.generateLabelsPushButton.clicked.connect(lambda: self.imagePreview.generateLabels())
        self.generateLabelsPushButton.setShortcut("Ctrl+L")

        layout = QGridLayout()
        layout.addWidget(self.showBoxesPushButton, 0, 0, 1, 1)
        layout.addWidget(self.generateLabelsPushButton, 1, 0, 1, 1)
        self.labelGroupBox.setLayout(layout)

    def createArrowKeysGroupBox(self):
        self.arrowKeysGroupBox = QGroupBox("")

        leftPushButton = QPushButton(self.arrowKeysGroupBox)
        leftPushButton.setText("Prev Img (Left)")
        leftPushButton.clicked.connect(lambda: self.prevImageCB())
        leftPushButton.setShortcut(Qt.Key_Left)

        rightPushButton = QPushButton(self.arrowKeysGroupBox)
        rightPushButton.setText("Next Img (Right)")
        rightPushButton.clicked.connect(lambda: self.nextImageCB())
        rightPushButton.setShortcut(Qt.Key_Right)

        upPushButton = QPushButton(self.arrowKeysGroupBox)
        upPushButton.setText("Zoom In (Up)")
        upPushButton.clicked.connect(lambda: self.imagePreview.zoomIn())
        upPushButton.setShortcut(Qt.Key_Up)

        downPushButton = QPushButton(self.arrowKeysGroupBox)
        downPushButton.setText("Zoom Out (Down)")
        downPushButton.clicked.connect(lambda: self.imagePreview.zoomOut())
        downPushButton.setShortcut(Qt.Key_Down)

        layout = QGridLayout()
        layout.addWidget(leftPushButton, 0, 0, 1, 1)
        layout.addWidget(rightPushButton, 0, 1, 1, 1)
        layout.addWidget(upPushButton, 1, 0, 1, 1)
        layout.addWidget(downPushButton, 1, 1, 1, 1)

        self.arrowKeysGroupBox.setLayout(layout)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    controls = LabelerControls()
    controls.show()
    controls.imagePreview.show()

    sys.exit(app.exec_())