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
        layout.addWidget(QLabel(text))
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)


class SimpleDialog(QDialog):
    """A very simple dialog that only displays one message
    (e.g. to instruct the user to wait)."""

    def __init__(self, text: str):
        super().__init__()

        layout = QVBoxLayout()
        layout.addWidget(QLabel(text))
        self.setLayout(layout)

