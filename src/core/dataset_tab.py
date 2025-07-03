from PyQt6.QtWidgets import QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox, QFileDialog
from PyQt6.QtCore import Qt


class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()

        self._create_ui()
        self.load_dataset_btn.clicked.connect(self._load_dataset_path)

    def _load_dataset_path(self):
        file_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        
        if file_path:
            self.info_label.setText(file_path)

    def _create_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)

        dataset_group = QGroupBox("Dataset Overview")
        dataset_group.setObjectName("dataset-overview")
        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(20)
        
        self.load_dataset_btn = QPushButton("Load Dataset")

        self.info_label = QLabel("Dataset information and statistics will be displayed here.")
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.info_label.setStyleSheet("color: gray; font-style: italic;")

        dataset_layout.addWidget(self.load_dataset_btn, 1)
        dataset_layout.addWidget(self.info_label, 4)
        dataset_group.setLayout(dataset_layout)

        main_layout.addWidget(dataset_group)
        main_layout.addStretch()