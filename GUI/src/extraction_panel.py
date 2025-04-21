from elecloc.main import extract_from_files

from utils import InfoDialog
from mediator_interface import MediatorInterface

from PyQt6.QtWidgets import (QWidget, 
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
    widget.textChanged.connect(callback)
    return widget


class ElecLocExtractionPanel(QWidget):

    #
    # Core
    #

    def __init__(self):
        super().__init__()

        self._updatable_children = {
            "inputCtPath" : None,
            "inputMaskPath" : None,
            "threshMin": None,
            "threshMax": None,

            "toggle_view_ct": None,
            "toggle_view_thresholded": None,
            "opacity_nifti": 0.2,
            "opactity_thresholded": 0.5,
        }

        # Adding the children widgets
        self._children = {}
        self._init_widgets()

    def _init_widgets(self) -> None:
        """Adds all the children widgets to this panel"""
        layout = QVBoxLayout(self)

        # Part 1: Core algorithm

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

        ## CT thresholds info
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
        # TODO ESSENTIAL link callback
        self.w_showCt.clicked.connect(lambda: None)
        layout_display_input.addWidget(self.w_showCt)

        self.w_opacityInput = QSlider(orientation=Qt.Orientation.Horizontal)
        # TODO ESSENTIAL link callback
        self.w_opacityInput.sliderReleased.connect(lambda: print("Released"))
        layout_display_input.addWidget(self.w_opacityInput)

        layout.addLayout(layout_display_input)

        # Showing and adjusting opacity of mask
        layout_display_mask = QHBoxLayout()

        self.w_showMask = QCheckBox("Show mask")
        # TODO ESSENTIAL link callback
        self.w_showMask.clicked.connect(lambda: None)
        layout_display_mask.addWidget(self.w_showMask)

        self.w_opacityMask = QSlider(orientation=Qt.Orientation.Horizontal)
        # TODO ESSENTIAL link callback
        self.w_opacityMask.sliderReleased.connect(lambda: print("Released"))
        layout_display_mask.addWidget(self.w_opacityMask)

        layout.addLayout(layout_display_mask)

        ## Button to launch extraction
        launch_button = QPushButton("Extract centroids")
        launch_button.clicked.connect(self._launch_extraction)
        layout.addWidget(launch_button)

        # Part 2: Adjustments
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

    def _launch_extraction(self) -> None:
        min_str = self.w_threshMin.text()
        max_str = self.w_threshMax.text()
        try:
            # TODO enhancement: progress bar
            centroids = extract_from_files(
                self.w_inputCtPath.text(),
                self.w_inputMaskPath.text(),
                int(min_str) if min_str != "" else None,
                int(max_str) if max_str != "" else None
            )
            self._mediator.set_centroids(centroids)
            dlg = InfoDialog("Success", f"Succesfully computed centroids.")
            
        except (ValueError, FileNotFoundError) as e:
            dlg = InfoDialog("Invalid argument", str(e))
            dlg.exec()

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

    #
    # Helper methods
    #

    def _linked_QLineEdit(
            self,
            param_name: str,
            placeholder: str = "",
            parent: QWidget = None
    ) -> QLineEdit:
        """Quick method to create a QLineEdit widget whose value is linked
        to the given parameter.
        
        ### Inputs:
        - param_name: the name of the parameters in 'self._params' linked to
        the value in this QLineEdit widget.
        - placeholder: the text to display when the widget is empty.
        
        ### Output:
        - widget: the specified QLineEdit."""
        # Widget
        widget = QLineEdit(parent)
        # Placeholder text
        widget.setPlaceholderText(placeholder)
        widget.displayText()
        # Callback
        def callback():
            self._params[param_name] = widget.text
        widget.textChanged.connect(callback)
        
        return widget

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
        
    