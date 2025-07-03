from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QStyle, QToolTip
from PyQt6.QtCore import Qt
import sys, os

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

        layout.addWidget(self.model_file_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addWidget(self.model_status_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addWidget(self.dataset_status_label)

    def set_model_file(self, file_name, color=None):
        if color is None:
            self.model_file_label.setText(f"Model File: {file_name}")
        else:
            self.model_file_label.setText(f'Model File: <span style="color: {color}">{file_name}</span>')

    def set_model_status(self, status, color=None):
        self.model_status_label.setText(status)
        if color is None:
            self.model_status_label.setStyleSheet("") 
        else:
            self.model_status_label.setStyleSheet(f"color: {color};")

    def set_dataset_status(self, status, color=None):
        self.dataset_status_label.setText(status)
        if color is None:
            self.dataset_status_label.setStyleSheet("")
        else:
            self.dataset_status_label.setStyleSheet(f"color: {color};")
      
    def get_project_root(self):
        if getattr(sys, 'frozen', False):
            # Executable
            return sys._MEIPASS
        else:
            # IDE
            current_file_path = os.path.abspath(__file__)
            widgets_dir = os.path.dirname(current_file_path)
            ui_dir = os.path.dirname(widgets_dir)
            src_dir = os.path.dirname(ui_dir)
            project_root = os.path.dirname(src_dir)
            return project_root