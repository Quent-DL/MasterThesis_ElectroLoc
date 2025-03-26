# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:03:28 2025

@author: nicol
"""

import sys
import numpy as np
import nibabel as nib
import pyvista as pv
from pyvistaqt import QtInteractor  # For embedding PyVista in PyQt6
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QFileDialog, QCheckBox, QSlider,
                             QLabel, QComboBox, QMainWindow, QLineEdit)
from dipy.io.streamline import load_tractogram
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class TrkViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.nb_electrodes = None
        self.intercontact_dist = None
        self.trk_file = None    # todo adapt to csv
        self.nii_file = None
        self.nii_data = None
        self.grid = None
        self.names = []
        self.selected_name = None
        self.X, self.Y, self.Y = None, None, None
        self.initUI()

    def initUI(self):
        # Create the main horizontal layout
        main_layout = QHBoxLayout(self)

        # Left side: Control panel
        control_layout = QVBoxLayout()

        self.labelTitle = QLabel()
        self.labelTitle.setText('Labellisation')
        control_layout.addWidget(self.labelTitle)

        self.widgetNbElectrodes = QLineEdit()
        self.widgetNbElectrodes.setPlaceholderText('Nb electrodes')
        self.widgetNbElectrodes.displayText()
        self.widgetNbElectrodes.textChanged.connect(self.set_nb_electrodes)
        control_layout.addWidget(self.widgetNbElectrodes)
        
        self.widgetIntercontactDist = QLineEdit()
        self.widgetIntercontactDist.setPlaceholderText('Distance')
        self.widgetIntercontactDist.displayText()
        self.widgetIntercontactDist.textChanged.connect(self.set_intercontact_dist)
        control_layout.addWidget(self.widgetIntercontactDist)

        self.loadButtonNifti = QPushButton('Load .nii.gz File')
        self.loadButtonNifti.clicked.connect(self.loadNiftiFile)
        control_layout.addWidget(self.loadButtonNifti)

        self.opacityLabel = QLabel('Opacity:')
        control_layout.addWidget(self.opacityLabel)

        self.opacitySlider = QSlider()
        # Vertical slider if needed; otherwise, default is horizontal
        # self.opacitySlider.setOrientation(1)
        self.opacitySlider.setMinimum(1)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(100)
        self.opacitySlider.sliderReleased.connect(self.viewPlot)
        control_layout.addWidget(self.opacitySlider)

        
        self.adjustTitle = QLabel()
        self.adjustTitle.setText('Adjustments')
        control_layout.addWidget(self.adjustTitle)

        self.widgetX = QLineEdit()
        self.widgetX.setPlaceholderText('x')
        self.widgetX.displayText()
        self.widgetX.textChanged.connect(self.set_coords)
        control_layout.addWidget(self.widgetX)

        self.widgetY = QLineEdit()
        self.widgetY.setPlaceholderText('y')
        self.widgetY.displayText()
        self.widgetY.textChanged.connect(self.set_coords)
        control_layout.addWidget(self.widgetY)
        
        self.widgetZ = QLineEdit()
        self.widgetZ.setPlaceholderText('Z')
        self.widgetZ.displayText()
        self.widgetZ.textChanged.connect(self.set_coords)
        control_layout.addWidget(self.widgetZ)

        self.showPointsCheckbox = QCheckBox('Show Points')
        control_layout.addWidget(self.showPointsCheckbox)

        self.colorMapLabel = QLabel('Color Map:')
        control_layout.addWidget(self.colorMapLabel)

        self.colorMapComboBox = QComboBox()
        self.colorMapComboBox.addItems(['flesh', 'Set3', 'tab20', 'plasma'])
        control_layout.addWidget(self.colorMapComboBox)

        self.viewButton = QPushButton('View 3D Plot')
        self.viewButton.clicked.connect(self.viewPlot)
        control_layout.addWidget(self.viewButton)

        # Add the control panel layout to the main layout
        main_layout.addLayout(control_layout)

        # Right side: PyVista viewer embedded in the GUI using QtInteractor
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor)

        self.setLayout(main_layout)
        self.setWindowTitle("Tractography Viewer")

    def set_coords(self):
        self.X = self.widgetX.displayText().strip()
        self.Y = self.widgetY.displayText().strip()
        self.Z = self.widgetZ.displayText().strip()

    def set_nb_electrodes(self):
        # TODO convert nb_electrodes to int when launching algo (currently str)
        self.nb_electrodes = self.widgetNbElectrodes.displayText().strip()

    def set_intercontact_dist(self):
        # TODO convert nb_electrodes to int when launching algo (currently str)
        self.intercontact_dist = self.widgetIntercontactDist.displayText().strip()

    # MODIFIED (added)
    # TODO implement
    def load_coords_from_CSV(self):
        # TODO put at top of file
        import pandas as pd
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open .csv file", "", "Comma Separated Values (*.csv)", options=options)
        df_coords = pd.read_csv(filePath, comment='#')

        self.df_coords = df_coords


    def loadNiftiFile(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open .nii.gz File", "", "Nifti Files (*.nii.gz)", options=options)
        if filePath:
            self.nii_file = filePath
            print(f"Loaded NIfTI file: {self.nii_file}")
            self.nii_data = nib.load(filePath).get_fdata()
            grid = pv.ImageData()
            grid.dimensions = np.array(self.nii_data.shape) + 1
            grid.cell_data['values'] = self.nii_data.flatten(order='F')
            self.grid = grid

    def callback(self, actor, myVar):
        print(actor)
        print(myVar)     # todo remove
        self.X = actor[0]
        self.Y = actor[1]
        self.Z = actor[2]
        self.widgetX.setText(str(round(self.X, 1)))
        self.widgetY.setText(str(round(self.Y, 1)))
        self.widgetZ.setText(str(round(self.Z, 1)))
        print(dir(myVar))   # TODO retrieve point id from attributes

    def viewPlot(self):

        # Clear previous meshes if necessary
        self.plotter.clear()

        opacity = self.opacitySlider.value() / 100.0
        show_points = self.showPointsCheckbox.isChecked()
        color_map = self.colorMapComboBox.currentText()

       
        # # Plot the tractography data on the embedded plotter
        # plot_trk(self.trk_file, opacity=opacity, plotter=self.plotter,
        #          show_points=show_points, color_map=color_map,
        #          name=self.trk_file, background='white')

        sphere = pv.Sphere()
        cube = pv.Cube().translate([10, 10, 0])

        points = self.df_coords[['ct_vox_x', 'ct_vox_y', 'ct_vox_z']].to_numpy()
        points_actor = self.plotter.add_mesh(
            points, color=(0, 0, 255), point_size=5.0, 
            render_points_as_spheres=True, pickable=True)
        
        if self.nii_data is not None and self.grid is not None:
            #     # Add a slice of the NIfTI volume to the scene
            #     self.plotter.add_mesh_slice(self.grid, cmap='gray',
            #                                 show_scalar_bar=False)

            self.plotter.add_volume(self.grid, cmap='gray', opacity=[0.0, 0.045],
                                    show_scalar_bar=False, pickable=False)

        sphere_actor = self.plotter.add_mesh(
            sphere, pickable=False)  # initially pickable
        cube_actor = self.plotter.add_mesh(
            cube, pickable=False)  # initially unpickable
        self.plotter.enable_point_picking(self.callback, show_message=False,
                                          show_point=True, use_picker=True)
        

        self.plotter.pickable_actors = [
            sphere_actor, cube_actor, points_actor]  # now both are pickable

        self.plotter.set_background('black')
        self.plotter.reset_camera()
        self.plotter.render()


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     viewer = TrkViewer()
#     viewer.show()
#     sys.exit(app.exec())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = TrkViewer()
        self.setCentralWidget(self.viewer)
        self.initMenuBar()
        self.setWindowTitle("Tractography Viewer with Menubar")

    def initMenuBar(self):
        menubar = self.menuBar()

        # File menu
        fileMenu = menubar.addMenu('File')

        # Action to load .trk file
        # MODIFIED
        loadCsvAction = QAction('Load .csv File', self)
        loadCsvAction.triggered.connect(self.viewer.load_coords_from_CSV)
        fileMenu.addAction(loadCsvAction)

        # Action to load .nii.gz file
        loadNiftiAction = QAction('Load .nii.gz File', self)
        loadNiftiAction.triggered.connect(self.viewer.loadNiftiFile)
        fileMenu.addAction(loadNiftiAction)

        # Optional: Exit action
        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())