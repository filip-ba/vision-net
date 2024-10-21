from PyQt6.QtWidgets import QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
from utils.image_loader import load_image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set the window title
        self.setWindowTitle("Fruit and Vegetable Recognition")

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout()

        # Create a button for loading an image
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        # Create a label to display the image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("No image loaded.")

        # Add button and label to layout
        layout.addWidget(self.load_button)
        layout.addWidget(self.image_label)

        # Set layout for central widget
        central_widget.setLayout(layout)

    def load_image(self):
        """Open a file dialog to select an image and display it."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = load_image(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setText("")
        else:
            self.image_label.setText("No image loaded.")
