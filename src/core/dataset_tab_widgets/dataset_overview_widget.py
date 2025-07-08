from PyQt6.QtWidgets import (QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt


class DatasetOverviewWidget(QWidget):

    def __init__(self, parent: QWidget | None):
        super().__init__(parent)
        self._create_ui()

    def set_path(self, path: str) -> None:
        self.dataset_path_label.setText(path)

    def reset(self) -> None:
        self.set_path("Dataset not loaded.")

    def _create_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        overview_group = QGroupBox("Dataset Overview")
        overview_group.setObjectName("dataset-overview")

        overview_layout = QHBoxLayout()
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(20)

        self.load_dataset_btn = QPushButton("Load Dataset")

        self.dataset_path_label = QLabel("Dataset not loaded.")
        self.dataset_path_label.setObjectName("dataset-path-label")
        self.dataset_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.dataset_path_label.setWordWrap(True)
        self.dataset_path_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        overview_layout.addWidget(self.load_dataset_btn, 1)
        overview_layout.addWidget(self.dataset_path_label, 4)
        overview_group.setLayout(overview_layout)

        main_layout.addWidget(overview_group)