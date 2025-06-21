from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox
from PyQt6.QtCore import Qt


class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        self._create_ui()

    def _create_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Dataset overview group box
        dataset_group = QGroupBox("Dataset Overview")
        dataset_group.setObjectName("dataset-group")
        dataset_layout = QVBoxLayout()
        
        # Placeholder content
        info_label = QLabel("Dataset information and statistics will be displayed here.")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: gray; font-style: italic;")
        
        dataset_layout.addWidget(info_label)
        dataset_group.setLayout(dataset_layout)

        main_layout.addWidget(dataset_group)
        main_layout.addStretch()