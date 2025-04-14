from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QStyle, QToolTip
from PyQt6.QtCore import Qt, QSize, QPoint
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
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
        
        self.help_icon = QToolButton()
        self.help_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion))
        self.help_icon.setIconSize(QSize(20, 20))
        self.help_icon.setStyleSheet("border: none; background-color: white;")  
        self.help_icon.setToolTip("Copy the 'fruitveg-dataset' folder to the 'dataset' folder in the project root directory.")
        self.help_icon.setVisible(False)  

        self.refresh_button = QPushButton()
        self.refresh_button.setMaximumSize(30, 30)
        self.refresh_button.setToolTip("Tries to load the dataset.")

        dataset_label_layout = QHBoxLayout()
        dataset_label_layout.setSpacing(3) 
        dataset_label_layout.addWidget(self.dataset_status_label)
        dataset_label_layout.addWidget(self.help_icon)

        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(0, 0, 20, 0)
        dataset_layout.addLayout(dataset_label_layout)  
        dataset_layout.addStretch()  
        dataset_layout.addWidget(self.refresh_button)

        layout.addWidget(self.model_file_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addWidget(self.model_status_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addLayout(dataset_layout)

        self.help_icon.clicked.connect(self._show_help_tooltip)

    def show_help_icon(self, show: bool):
        self.help_icon.setVisible(show)

    def _show_help_tooltip(self):
        QToolTip.showText(
            self.help_icon.mapToGlobal(QPoint(0, self.help_icon.height())),
            self.help_icon.toolTip(),
            self.help_icon
        )

    def set_model_file(self, file_name, color=None):
        """Updates the model file label with optional color for the file name."""
        if color is None:
            self.model_file_label.setText(f"Model File: {file_name}")
        else:
            self.model_file_label.setText(f'Model File: <span style="color: {color}">{file_name}</span>')

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

    def update_refresh_icon(self, theme: str):
        """Updates the refresh icon based on the current theme."""
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

        icon_path = os.path.join(project_root, "assets", "icons", f"refresh-{theme}.png")

        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            self.refresh_button.setIcon(QIcon(pixmap))
            self.refresh_button.setIconSize(QSize(20, 20))