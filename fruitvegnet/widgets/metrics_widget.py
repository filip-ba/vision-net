from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame


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
        layout.addWidget(self.create_separator())
        layout.addWidget(self.precision_label)
        layout.addWidget(self.create_separator())
        layout.addWidget(self.recall_label)

        self.setLayout(layout)

    def create_separator(self):
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setLineWidth(0)
        separator.setMidLineWidth(0)
        separator.setStyleSheet(
            "color: #e3e3e3; "
            "height: 1px; "
            "margin: 0px; "
            "padding: 0px;"
        )
        return separator

    def reset_metrics(self):
        self.accuracy_label.setText("Accuracy: -")
        self.precision_label.setText("Precision: -")
        self.recall_label.setText("Recall: -")

    def update_metrics(self, metrics):
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.0%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.0%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.0%}")