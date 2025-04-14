from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

from ...utils.custom_separator import create_separator


class ModelInfoWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._create_ui()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        self.model_file_label = QLabel("Model File: None")
        self.model_file_label.setObjectName("ModelFileLabel")
        self.model_file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)

        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setObjectName("ModelStatus")
        self.model_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)

        self.dataset_status_label = QLabel("No dataset found")
        self.dataset_status_label.setObjectName("ModelStatus")
        self.dataset_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)

        self.refresh_button = QPushButton()
        self.refresh_button.setMaximumSize(30, 30)

        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(0, 0, 20, 0)
        dataset_layout.addWidget(self.dataset_status_label)
        dataset_layout.addStretch
        dataset_layout.addWidget(self.refresh_button)

        layout.addWidget(self.model_file_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addWidget(self.model_status_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addLayout(dataset_layout)

    def set_model_file(self, file_name):
        """Updates the model file label."""
        self.model_file_label.setText(f"Model File: {file_name}")

    def set_model_status(self, status, color=None):
        """Updates the model status label."""
        self.model_status_label.setText(status)
        if color is None:
            self.model_status_label.setStyleSheet("") 
        else:
            self.model_status_label.setStyleSheet(f"color: {color};")

    def set_dataset_status(self, status, color=None):
        """Updates the dataset status label."""
        self.dataset_status_label.setText(status)
        if color is None:
            self.dataset_status_label.setStyleSheet("")
        else:
            self.dataset_status_label.setStyleSheet(f"color: {color};")