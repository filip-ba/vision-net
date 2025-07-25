from PyQt6.QtWidgets import ( QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, 
                              QLabel, QSizePolicy, QPushButton )
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
import os

from ...utils.custom_separator import create_separator
from ...utils.get_project_root import get_project_root


class DatasetStatusWidget(QWidget):
    
    def __init__(self, parent: QWidget | None, model_tabs: list):
        super().__init__(parent)
        self.model_tabs = model_tabs
        self.status_labels: dict[str, QLabel] = {}
        self._create_ui()

    def set_status(self, model_name: str, text: str, color: str = "black"):
        lbl = self.status_labels.get(model_name)
        if lbl is not None:
            lbl.setText(text)
            lbl.setStyleSheet(f"color: {color};")

    def reset(self):
        for lbl in self.status_labels.values():
            lbl.setText("-")
            lbl.setStyleSheet("color: gray;")

    def _create_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        status_group = QGroupBox("Dataset Status")
        status_group.setObjectName("dataset-status")
        
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(18, 0, 18, 0)
        status_layout.setSpacing(18)

        for idx, tab in enumerate(self.model_tabs):
            col_layout = QVBoxLayout()
            col_layout.setSpacing(4)
            col_layout.setContentsMargins(0, 4, 0, 4)

            name_lbl = QLabel(tab.model_name)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            status_lbl = QLabel("-")
            status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_lbl.setWordWrap(True)
            status_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
            status_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            col_layout.addWidget(name_lbl)
            col_layout.addWidget(status_lbl)
            status_layout.addLayout(col_layout)

            self.status_labels[tab.model_name] = status_lbl

            if idx != len(self.model_tabs) - 1:
                separator = create_separator("vertical")
                status_layout.addWidget(separator)

        status_group.setLayout(status_layout)

        self.check_status_btn = QPushButton()
        self.check_status_btn.setObjectName("refresh-button")
        project_root = get_project_root()
        icon_path = os.path.join(project_root, "assets", "icons", "refresh-dark.png")
        self.check_status_btn.setIcon(QIcon(icon_path))
        self.check_status_btn.setIconSize(QSize(24, 24))
        self.check_status_btn.setFixedSize(QSize(32, 32))
        self.check_status_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        main_layout.addWidget(status_group) 
        
        button_container = QWidget()
        button_container.setFixedWidth(52)  
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 28, 25, 0)  
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.check_status_btn)
        
        main_layout.addWidget(button_container)