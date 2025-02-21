from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os


class ImageClassificationWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Controls area with buttons
        controls_layout = QVBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.setStyleSheet("padding: 8px;")
        self.load_image_btn.setFixedWidth(120)
        self.classify_btn = QPushButton("Classify Image")
        self.classify_btn.setFixedWidth(120)
        self.classify_btn.setStyleSheet("padding: 8px;")
        self.result_label = QLabel("Classification:\nNone")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFixedWidth(120)
        self.result_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                padding: 1px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)

        # Small image preview
        self.image_display = QLabel()
        self.image_display.setFixedSize(120, 120)
        self.image_title = QLabel("Image Preview")
        self.image_display.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.load_image_btn)
        controls_layout.addWidget(self.classify_btn)
        controls_layout.addWidget(self.image_display, alignment=Qt.AlignmentFlag.AlignHCenter)
        controls_layout.addWidget(self.result_label)

        # Classification plot
        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(300, 300) 
        self.init_plot()

        layout.addWidget(self.canvas)
        layout.addLayout(controls_layout)

        # Load placeholder image
        self._load_placeholder()

    def init_plot(self):
        """Initialize empty probability plot"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title('Class Probabilities', fontsize=10)
        ax.set_xlabel('Class', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_ylim(0, 1)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot(self, classes, probabilities):
        """Update probability plot with new data"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        bars = ax.bar(classes, probabilities)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0%}',
                   ha='center', va='bottom')
            
        ax.set_title('Class Probabilities', fontsize=10)
        ax.set_xlabel('Class', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_ylim(0, 1.2) 
        plt.setp(ax.get_xticklabels(), rotation=45)
        self.figure.tight_layout()
        self.canvas.draw()

    def _load_placeholder(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        placeholder_path = os.path.join(project_root, "assets", "placeholder_img.png")

        if os.path.exists(placeholder_path):
            pixmap = QPixmap(placeholder_path)
            self.image_display.setPixmap(pixmap.scaled(
                120, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.image_display.setText("No image")

    def update_image(self, pixmap):
        """Method for updating the displayed image"""
        if pixmap:
            scaled_pixmap = pixmap.scaled(
                120, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
        else:
            self.image_display.setText("No image")