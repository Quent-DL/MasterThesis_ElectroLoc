from extraction_panel import ElecLocExtractionPanel
from interactive_plotter import InteractivePlotter
from mediator import Mediator

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
        self.w_extraction_panel = ElecLocExtractionPanel()
        biviewer.addWidget(self.w_extraction_panel)
        self._interactive_plotter = InteractivePlotter()  
        biviewer.addWidget(self._interactive_plotter.get_widget())
        self.setLayout(biviewer)

        # Adding a mediator to allow indirect and coordinated communication
        # between the panel extraction and the viewer
        self.mediator = Mediator(self.w_extraction_panel,
                                 self._interactive_plotter)
        self.w_extraction_panel.add_mediator(self.mediator)
        self._interactive_plotter.add_mediator(self.mediator)


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