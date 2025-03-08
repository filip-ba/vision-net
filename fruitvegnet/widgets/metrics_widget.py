from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt
from utils.custom_separator import create_separator


class MetricsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(5, 5, 5, 5)

        # Metrics grid for accuracy, precision, recall
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)

        self.accuracy_label = QLabel("Accuracy: -")
        self.precision_label = QLabel("Precision: -")
        self.recall_label = QLabel("Recall: -")

        metric_style = """
            QLabel {
                font-size: 14px;
                padding: 8px;
                background-color: #f0f0f0;
                border: 1px solid #bbb;
                border-radius: 5px;
                font-weight: 600;
            }
        """
        for label in [self.accuracy_label, self.precision_label, self.recall_label]:
            label.setStyleSheet(metric_style)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
        metrics_grid.addWidget(self.accuracy_label, 0, 0)
        metrics_grid.addWidget(self.precision_label, 0, 1)
        metrics_grid.addWidget(self.recall_label, 0, 2)
        metrics_grid.setSpacing(10)
        layout.addLayout(metrics_grid)


    def reset_metrics(self):
        self.accuracy_label.setText("Accuracy: -")
        self.precision_label.setText("Precision: -")
        self.recall_label.setText("Recall: -")

    def update_metrics(self, metrics):
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.2%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.2%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.2%}")