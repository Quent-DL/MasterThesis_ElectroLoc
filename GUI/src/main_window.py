from extraction_panel import ElecLocExtractionPanel
from interactive_plotter import InteractivePlotter
from misc.mediator import Mediator
from misc.data_center import DataCenter

from PyQt6.QtWidgets import QWidget, QMainWindow, QDialog, QHBoxLayout
from PyQt6.QtGui import QAction

import numpy as np


class BiViewer(QWidget):
    """This class defines a widget that show a control panel (on the left)
    and a PyVista viewer (on the right). It also ensures all communication
    between the two (i.e. the control panel defines what to display in the viewer,
    and interacting with the viewer affects the control panel)."""

    def __init__(self, data_center: DataCenter):
        super().__init__()

        self._data_center = data_center

        self._initUI()

    def _initUI(self):
        biviewer = QHBoxLayout()
        self.w_extraction_panel = ElecLocExtractionPanel()
        biviewer.addWidget(self.w_extraction_panel)
        self._interactive_plotter = InteractivePlotter()  
        biviewer.addWidget(self._interactive_plotter.get_widget())
        self.setLayout(biviewer)

        # Adding a mediator to allow indirect and coordinated communication
        # between the panel extraction and the viewer
        self.mediator = Mediator(self.w_extraction_panel,
                                 self._interactive_plotter,
                                 self._data_center)
        self.w_extraction_panel.add_mediator(self.mediator)
        self._interactive_plotter.add_mediator(self.mediator)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # An object to conveniently store information 
        # (centroids, ct, electrodes, ...) across the different
        # widgets of the app.
        data_center = DataCenter()

        self.biviewer = BiViewer(data_center)
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