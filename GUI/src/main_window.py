from extraction_panel import ElecLocExtractionPanel
#from interactive_plotter import InteractivePlotter

from PyQt6.QtWidgets import QWidget, QMainWindow, QDialog, QHBoxLayout
from PyQt6.QtGui import QAction
from pyvistaqt import QtInteractor

import numpy as np


class BiViewer(QWidget):
    """This class defines a widget that show a control panel (on the left)
    and a PyVista viewer (on the right). It also ensures all communication
    between the two (i.e. the control panel defines what to display in the viewer,
    and interacting with the viewer affects the control panel)."""

    def __init__(self):
        super().__init__()
        self._initUI()

        self._centroids = np.empty((0,3))

    def _initUI(self):
        biviewer = QHBoxLayout()
        self.w_extraction_panel = ElecLocExtractionPanel(self)
        biviewer.addWidget(self.w_extraction_panel)
        self.w_plotter = QtInteractor()  # TODO replace by custom wrapper below
        #self.w_plotter = InteractivePlotter(self)   
        biviewer.addWidget(self.w_plotter)
        self.setLayout(biviewer)

    def update_centroids(self, centroids: np.ndarray) -> None:
        self._centroids = centroids

        # TODO plot centroids in self.w_plotter


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.biviewer = BiViewer()
        self.setCentralWidget(self.biviewer)
        self.initMenuBar()
        self.setWindowTitle("ElecLoc")

    def initMenuBar(self):
        menubar = self.menuBar()

        # File menu
        fileMenu = menubar.addMenu('File')

        # Optional: Exit action
        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

    def update_centroids(self, centroids: np.ndarray) -> None:
        self._centroids = centroids
        # TODO display change in viewer
