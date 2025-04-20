from elecloc.main import extract_from_files

from utils import InfoDialog

from PyQt6.QtWidgets import (QWidget, 
                             QHBoxLayout, QVBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox)
from PyQt6.QtGui import QIntValidator
from typing import Callable

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

    def __init__(self, parent_biviewer):
        super().__init__()

        self.parent_biviewer = parent_biviewer

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
        input_layout = QHBoxLayout()

        self.w_inputCtPath = create_QLineEdit("e.g.: ./input_CT.nii.gz")
        input_layout.addWidget(self.w_inputCtPath)

        input_button = QPushButton("Browse")
        input_button.clicked.connect(
            lambda: self._browse_file(self.w_inputCtPath))
        input_layout.addWidget(input_button)

        layout.addLayout(input_layout)

        layout.addWidget(QLabel(text="Path to mask:"))
        input_mask_layout = QHBoxLayout()

        self.w_inputMaskPath = create_QLineEdit("e.g.: ./mask.nii.gz")
        input_mask_layout.addWidget(self.w_inputMaskPath)

        input_mask_button = QPushButton("Browse")
        input_mask_button.clicked.connect(
            lambda: self._browse_file(self.w_inputMaskPath))
        input_mask_layout.addWidget(input_mask_button)

        layout.addLayout(input_mask_layout)

        ## CT thresholds info
        layout.addWidget(QLabel(text="Thresholds (leave empty for automatic computation):"))
        thresh_layout = QHBoxLayout()

        thresh_layout.addWidget(QLabel("Min:"))

        self.w_threshMin = QLineEdit()
        self.w_threshMin.setValidator(QIntValidator())
        thresh_layout.addWidget(self.w_threshMin)
        self.w_threshMin.setPlaceholderText("[Compute]")
        self.w_threshMin.setText("1500")

        thresh_layout.addWidget(QLabel("Max:"))

        self.w_threshMax = QLineEdit()
        self.w_threshMax.setValidator(QIntValidator())
        thresh_layout.addWidget(self.w_threshMax)
        self.w_threshMax.setPlaceholderText("None")

        layout.addLayout(thresh_layout)

        ## Button to launch extraction
        launch_button = QPushButton("Extract centroids")
        launch_button.clicked.connect(self._launch_extraction)
        layout.addWidget(launch_button)

        # Part 2: Adjustments

        ## Adding new centroid

        ## Selecting/deleting centroid

        ## Modifying position of centroid

        # Part 3: buttons to save results as CSV

        ...

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
            self.parent_biviewer.update_centroids(centroids)

            dlg = InfoDialog("Success", f"Succesfully computed centroids.")

            # TODO incomplete: replace by actually useful code
            print(centroids)
        except (ValueError, FileNotFoundError) as e:
            dlg = InfoDialog("Invalid argument", str(e))
            dlg.exec()


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