from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt

from ...utils.custom_separator import create_separator


class MetricsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        self.accuracy_label = QLabel("Accuracy: -")
        self.accuracy_label.setObjectName("MetricsAccuracy")
        self.accuracy_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        self.precision_label = QLabel("Precision: -")
        self.precision_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        self.recall_label = QLabel("Recall: -")
        self.recall_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)

        layout.addWidget(self.accuracy_label)
        layout.addWidget(create_separator("vertical"))
        layout.addWidget(self.precision_label)
        layout.addWidget(create_separator("vertical"))
        layout.addWidget(self.recall_label)

        self.setLayout(layout)

    def reset_metrics(self):
        self.accuracy_label.setText("Accuracy: -")
        self.precision_label.setText("Precision: -")
        self.recall_label.setText("Recall: -")
        self.recall_label.setToolTip("")

    def update_metrics(self, metrics):
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.0%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.0%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.0%}")
        
        if 'class_recall' in metrics and 'class_names' in metrics:
            tooltip_text = "Per-class Recall:\n"
            for i, (class_name, recall_value) in enumerate(zip(metrics['class_names'], metrics['class_recall'])):
                tooltip_text += f"{class_name}: {recall_value:.0%}\n"
            self.recall_label.setToolTip(tooltip_text)