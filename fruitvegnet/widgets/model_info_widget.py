from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from utils.custom_separator import create_separator


class ModelInfoWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._create_ui()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        # Model status label
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setObjectName("ModelStatus")
        layout.addWidget(self.model_status_label)

        layout.addWidget(create_separator())

        # Model file label
        self.model_file_label = QLabel("Model File: None")
        self.model_file_label.setObjectName("ModelFileLabel")
        layout.addWidget(self.model_file_label)

    def set_model_status(self, status, color="red"):
        """Updates the model status label."""
        self.model_status_label.setText(f"{status}")
        self.model_status_label.setStyleSheet(f"color: {color};")

    def set_model_file(self, file_name):
        """Updates the model file label."""
        self.model_file_label.setText(f"Model File: {file_name}")