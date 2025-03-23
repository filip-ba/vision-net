from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

from src.utils.custom_separator import create_separator


class MetricsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        self.accuracy_label = QLabel("Accuracy: -")
        self.accuracy_label.setObjectName("MetricsAccuracy")
        self.precision_label = QLabel("Precision: -")
        self.recall_label = QLabel("Recall: -")

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

    def update_metrics(self, metrics):
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.0%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.0%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.0%}")