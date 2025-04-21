from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel

class InfoDialog(QDialog):
    """A simple class for displaying informative dialog which only consist in
    a window title, some text, and a button "OK"."""
    def __init__(self, title: str, text: str):
        super().__init__()

        self.setWindowTitle(title)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.accepted.connect(self.accept)

        layout = QVBoxLayout()
        message = QLabel(text)
        layout.addWidget(message)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
