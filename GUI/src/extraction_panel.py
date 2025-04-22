from elecloc.main import extract_from_files

from misc.dialogs import InfoDialog, SimpleDialog
from misc.mediator_interface import MediatorInterface

from PyQt6.QtWidgets import (QWidget, QDialog,
                             QHBoxLayout, QVBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QFileDialog, 
                             QDoubleSpinBox, QCheckBox, QSlider)
from PyQt6.QtGui import QIntValidator, QIcon
from PyQt6.QtCore import Qt
import numpy as np
from typing import Callable
import os


def get_icon_src(icon_name) -> str:
    ICON_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
    return os.path.join(ICON_DIR, icon_name)


def create_QLineEdit(
        placeholder: str = "", 
        callback: Callable[[], None] = lambda: None,
        parent: QWidget = None
) -> QLineEdit:
    """Wrapper to quickly create a QLineEdit widget with a given placeholder
    and callback function.
    
    ### Inputs:
    - placeholder: the text to display by default. Default: no text.
    - callback: the function to call when the value of the QLineEdit field
    is modified. Doesn't take any argument or return anything. Default: no
    callback.
    - parent: the parent widget
    
    ### Outputs:
    - widget: the specified QLineEdit widget."""
    widget = QLineEdit(parent)
    widget.setPlaceholderText(placeholder)
    widget.displayText()
    widget.editingFinished.connect(callback)
    return widget


class ElecLocExtractionPanel(QWidget):

    #
    # Core
    #

    def __init__(self):
        super().__init__()

        # Adding the children widgets
        self._children = {}
        self._init_widgets()

    def _init_widgets(self) -> None:
        """Adds all the children widgets to this panel"""
        layout = QVBoxLayout(self)

        # Part 1: Loading CT

        ## Input Nifti file and mask info
        layout.addWidget(QLabel(text="Path to input Nifti file [*]:"))
        layout_input = QHBoxLayout()

        self.w_inputCtPath = create_QLineEdit("e.g.: ./input_CT.nii.gz")
        layout_input.addWidget(self.w_inputCtPath)

        # TODO enhancement: autofill min and max thresholds placeholders
        input_button = QPushButton("Browse")
        input_button.clicked.connect(
            lambda: self._browse_file(self.w_inputCtPath))
        layout_input.addWidget(input_button)

        layout.addLayout(layout_input)

        layout.addWidget(QLabel(text="Path to mask:"))
        layout_input_mask = QHBoxLayout()

        self.w_inputMaskPath = create_QLineEdit("e.g.: ./mask.nii.gz")
        layout_input_mask.addWidget(self.w_inputMaskPath)

        input_mask_button = QPushButton("Browse")
        input_mask_button.clicked.connect(
            lambda: self._browse_file(self.w_inputMaskPath))
        layout_input_mask.addWidget(input_mask_button)

        layout.addLayout(layout_input_mask)

        self.w_buttonLoadCt = QPushButton("Load files")
        self.w_buttonLoadCt.clicked.connect(self._load_ct)
        layout.addWidget(self.w_buttonLoadCt)

        ## Part 2: CT thresholds info and extraction launcher
        self.w_ctInfo = InformationCT()
        self.w_ctInfo.setVisible(False)
        layout.addWidget(self.w_ctInfo)

        # Part 3: Adjustments
        self.w_centroid_menu = CentroidInfoPanel()
        self.w_centroid_menu.setVisible(False)
        layout.addWidget(self.w_centroid_menu)

        # Part 3: buttons to save results as CSV
        # TODO ESSENTIAL
        ...

        layout.addStretch()

    def add_mediator(self, mediator: MediatorInterface) -> None:
        self._mediator = mediator
        # Propagates the addition of the mediator to the centroid menu,
        # to define its callbacks
        self.w_centroid_menu.setup_callbacks(mediator)
        self.w_ctInfo.setup_callbacks(mediator)

    #
    # Callback methods
    #

    def _browse_file(self, widget: QLineEdit) -> None:
        """Browse a file and modifies the appropriate QLineEdit widget
        to display the name of the file selected.

        To use this function as a callback, use the following syntax
        with the appropriate parameter name:
        lambda: self._browse_file(widget)
        
        ### Inputs:
        - paramName: the name of the QLabel widget to update in 
        self._updatable_children."""
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open .nii.gz File", "", 
            "Nifti Files (*.nii.gz)")
        if filePath != "":
            widget.setText(filePath)

    def _load_ct(self) -> None:
        """Loads the CT files in the QEditLine widgets."""
        ct_path = self.w_inputCtPath.text().strip()
        mask_path = self.w_inputMaskPath.text().strip()

        # Temporarily disabling button
        self.w_buttonLoadCt.setEnabled(False)
        old_text = self.w_buttonLoadCt.text()
        self.w_buttonLoadCt.setText("Loading files, wait...")

        # TODO enhancement: loading dialog

        success = self._mediator.load_plot_ct_volumes(ct_path, mask_path)
        if success:
            self.w_ctInfo.setVisible(True)
        # Do nothing if load failed

        self.w_buttonLoadCt.setEnabled(True)
        self.w_buttonLoadCt.setText(old_text)

    #
    # API for mediator
    #

    def display_selected_centroid(self, coords: np.ndarray) -> None:
        """Displays in the centroid menu the given coordinates. Shape (3,).
        The coordinates are rounded to their 3rd decimals."""
        x, y, z = np.round(coords, 3)
        self.w_centroid_menu.w_centroid_x.setValue(x)
        self.w_centroid_menu.w_centroid_y.setValue(y)
        self.w_centroid_menu.w_centroid_z.setValue(z)
        self.w_centroid_menu.setVisible(True)

    def unselect(self) -> None:
        """Defines the behaviour for when no centroid is selected. In this
        implementation, the centroid menu is simply hidden."""
        self.w_centroid_menu.setVisible(False)


        
class InformationCT(QWidget):
    """A widget with various children to modify the display of the CT."""

    def __init__(self):
        super().__init__()
        self._init_UI()

    def _init_UI(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(text="Thresholds (leave empty for automatic computation):"))
        layout_thresh = QHBoxLayout()

        layout_thresh.addWidget(QLabel("Min:"))

        self.w_threshMin = QLineEdit()
        self.w_threshMin.setValidator(QIntValidator())
        layout_thresh.addWidget(self.w_threshMin)
        self.w_threshMin.setPlaceholderText("[Compute]")
        self.w_threshMin.setText("1500")

        layout_thresh.addWidget(QLabel("Max:"))

        self.w_threshMax = QLineEdit()
        self.w_threshMax.setValidator(QIntValidator())
        layout_thresh.addWidget(self.w_threshMax)
        self.w_threshMax.setPlaceholderText("None")

        layout.addLayout(layout_thresh)

        # Showing and adjusting opacity of CT scan
        layout_display_input = QHBoxLayout()

        self.w_showCt = QCheckBox("Show CT")
        self.w_showCt.setChecked(True)
        layout_display_input.addWidget(self.w_showCt)

        self.w_opacityInput = InformationCT.OpacitySlider()
        layout_display_input.addWidget(self.w_opacityInput)

        layout.addLayout(layout_display_input)

        # Showing and adjusting opacity of mask
        layout_display_mask = QHBoxLayout()

        self.w_showMask = QCheckBox("Show mask")
        layout_display_mask.addWidget(self.w_showMask)

        self.w_opacityMask = InformationCT.OpacitySlider()
        layout_display_input.addWidget(self.w_opacityMask)

        layout.addLayout(layout_display_mask)

        ## Button to launch extraction
        self.w_launchButton = QPushButton("Extract centroids")
        layout.addWidget(self.w_launchButton)

        # Global layout
        layout.addStretch()

    def setup_callbacks(self, mediator: MediatorInterface) -> None:
        """Adds a callback to all the necessary widgets using the given
        mediator object."""
    
        def _launch_extraction() -> None:
            min_str = self.w_threshMin.text()
            threshMin = int(min_str) if min_str != "" else None
            max_str = self.w_threshMax.text()
            threshMax = int(max_str) if max_str != "" else None

            self.w_launchButton.setEnabled(False)
            old_text = self.w_launchButton.text()
            self.w_launchButton.setText("Computing, wait...")

            try:
                # TODO enhancement: progress bar

                centroids = extract_from_files(
                    self.w_inputCtPath.text(),
                    self.w_inputMaskPath.text(),
                    threshMin,
                    threshMax
                )
                mediator.set_centroids(centroids)
                dlg_success = InfoDialog("Success", f"Succesfully computed centroids.")
                dlg_success.exec()
                
            except (ValueError, FileNotFoundError) as e:
                dlg_fail = InfoDialog("Invalid argument", str(e))
                dlg_fail.exec()

            self.w_launchButton.setEnabled(True)
            self.w_launchButton.setText(old_text)

        def _update_ct_options():
            visibilityCt = self.w_showCt.isChecked()
            opacityCt = self.w_opacityInput.value()
            mediator.update_ct_display(visibilityCt, opacityCt)

        def _replot_thresholded():
            min_str = self.w_threshMin.text()
            threshMin = int(min_str) if min_str != "" else None
            max_str = self.w_threshMax.text()
            threshMax = int(max_str) if max_str != "" else None

            visibilityMask = self.w_showMask.isChecked()
            opacityMask = self.w_opacityMask.value()
            mediator.plot_thresholded_volume(
                threshMin, threshMax, visibilityMask, opacityMask)
        
        def _update_thresholded_options():
            mediator.update_thresholded_display(
            visibility = self.w_showMask.isChecked(),
            opacity = self.w_opacityMask.value())

        self.w_launchButton.clicked.connect(_launch_extraction)

        # Modifying the thresholds
        self.w_threshMin.editingFinished.connect(_replot_thresholded)
        self.w_threshMax.editingFinished.connect(_replot_thresholded)

        # Toggle view volumes
        self.w_showCt.checkStateChanged.connect(_update_ct_options)
        self.w_showMask.checkStateChanged.connect(_update_thresholded_options)

        # Opacity volumes
        self.w_opacityInput.sliderReleased.connect(_update_ct_options)
        self.w_opacityMask.sliderReleased.connect(_update_thresholded_options)

    class OpacitySlider(QSlider):
        """A slider to define a floating point value opacity"""
        NB_STEPS = 20

        def __init__(self):
            super().__init__()
            self.setOrientation(Qt.Orientation.Horizontal)
            self.setRange(0, self.NB_STEPS)
            self.setTickInterval(1)
            self.setSingleStep(1)
            self.setSliderPosition(10)
    
        # Override
        def value(self) -> float:
            return super().value() / self.NB_STEPS
        
        
class CentroidInfoPanel(QWidget):
    """This widget allows the user to display information about a set of
    centroids, as well as modify them."""

    def __init__(self):
        super().__init__()
        self._init_UI()

    def _init_UI(self) -> None:
        layout = QVBoxLayout(self)

        # Label and "add centroid" button
        layout_title = QHBoxLayout()

        layout_title.addWidget(QLabel("Centroid Menu"))

        self.w_add_centroid = QPushButton()
        self.w_add_centroid.setIcon(QIcon(get_icon_src("add")))
        layout_title.addWidget(self.w_add_centroid)

        layout.addLayout(layout_title)

        # Centroids drop-down and delete button
        layout_selection = QHBoxLayout()

        self.w_delete_centroid = QPushButton()
        self.w_delete_centroid.setIcon(QIcon(get_icon_src("trash")))
        layout_selection.addWidget(self.w_delete_centroid)

        layout.addLayout(layout_selection)

        # Coordinates of the selected centroid
        layout_coords = QHBoxLayout()
        
        # X
        layout_coords.addWidget(QLabel("x"))
        self.w_centroid_x = self._create_coords_box()
        layout_coords.addWidget(self.w_centroid_x)
        # Y
        layout_coords.addWidget(QLabel("y"))
        self.w_centroid_y = self._create_coords_box()
        layout_coords.addWidget(self.w_centroid_y)
        # Z
        layout_coords.addWidget(QLabel("z"))
        self.w_centroid_z = self._create_coords_box()
        layout_coords.addWidget(self.w_centroid_z)

        layout.addLayout(layout_coords)
    
    def setup_callbacks(self, mediator: MediatorInterface) -> None:
        """Adds a callback to all the necessary widgets using the given
        mediator object."""

        def update_coords() -> None:
            """Notifies the mediator that the coordinates of the selected centroid
            have been modified."""
            x = self.w_centroid_x.value()
            y = self.w_centroid_y.value()
            z = self.w_centroid_z.value()

            coords = np.array([x, y, z], dtype=np.float32)
            mediator.update_selected_centroid(coords)

        self.w_add_centroid.released.connect(mediator.add_centroid)
        self.w_delete_centroid.released.connect(
            mediator.delete_selected_centroid)
        
        self.w_centroid_x.valueChanged.connect(update_coords)
        self.w_centroid_y.valueChanged.connect(update_coords)
        self.w_centroid_z.valueChanged.connect(update_coords)

    def _create_coords_box(self) -> QDoubleSpinBox:
        """Quickly creates a float quickbox with a callback to the mediator
        when the value is changed"""
        widget = QDoubleSpinBox()
        widget.setRange(-1e9, 1e9)
        widget.setSingleStep(1.0)
        return widget
        