from PyQt6.QtWidgets import QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.create_ui()
        # Connects
        self.btn_select_img.clicked.connect(self.open_image_dialog)

    def create_ui(self):
        self.setWindowTitle("FruitVegNet")
        self.setGeometry(100, 100, 600, 400)
        
        main_widget = QWidget()
        layout = QVBoxLayout()

        self.label = QLabel("Upload an image of fruits/vegetables", self)
        self.btn_select_img = QPushButton("Select Image", self)

        layout.addWidget(self.label)
        layout.addWidget(self.btn_select_img)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def open_image_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg);;All Files (*)")
        if file_name:
            self.label.setText(f"Selected: {file_name}")
