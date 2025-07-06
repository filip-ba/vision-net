from PyQt6.QtWidgets import ( QWidget, QGroupBox, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSizePolicy )
from PyQt6.QtCore import Qt

from ...utils.custom_separator import create_separator


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
        main_layout = QVBoxLayout(self)
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
        main_layout.addWidget(status_group)
        main_layout.addStretch()