from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QGroupBox, QVBoxLayout, 
    QFileDialog, QLabel, QPushButton, QHBoxLayout )
from PyQt6.QtGui import QIcon, QPixmap, QImageReader, QAction
from PyQt6.QtCore import Qt, pyqtSignal
import os
import random


class ImageClassificationWidget(QWidget):
    image_loaded = pyqtSignal(str)  # Signal to notify when new image is loaded
    classify_clicked = pyqtSignal()  # Signal for classify button clicks
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self._create_ui()
        
    def _create_ui(self):
        # Create main GroupBox
        group_box = QGroupBox("Image Classification")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #c4c8cc;
                border-radius: 6px;
                margin-top: 20px;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 3px 0 3px;
            }
        """)
        
        # Main layout inside GroupBox
        main_layout = QHBoxLayout(group_box)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image preview
        self.image_display = QLabel()
        self.image_display.setFixedSize(120, 120)
        self.image_display.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Buttons
        button_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.classify_btn = QPushButton("Classify All")
        for btn in [self.load_btn, self.classify_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    padding: 8px;
                    font-size: 14px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
            """)
            button_layout.addWidget(btn)
        
        # Combined results label
        self.results_label = QLabel("Classification Results:\n\n"
                                  "Simple CNN: None\n"
                                  "ResNet: None\n"
                                  "EfficientNet: None\n"
                                  "VGG16: None")
        self.results_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                min-width: 200px;
            }
        """)
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Layout setup
        controls_layout = QVBoxLayout()
        controls_layout.addLayout(button_layout)
        controls_layout.addWidget(self.results_label)
        controls_layout.addStretch()
        
        main_layout.addWidget(self.image_display)
        main_layout.addLayout(controls_layout)
        main_layout.addStretch()
        
        # Set the GroupBox as the main widget layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(group_box)
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_image)
        self.classify_btn.clicked.connect(self.classify_clicked.emit)
        
        # Load initial image from dataset
        self.load_random_test_image()
        
    def load_random_test_image(self):
        """Load a random image from the test dataset"""
        dataset_path = "./dataset/fruit_dataset/test"
        if os.path.exists(dataset_path):
            # Get all class directories
            class_dirs = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
            
            if class_dirs:
                # Choose random class
                random_class = random.choice(class_dirs)
                class_path = os.path.join(dataset_path, random_class)
                
                # Get all images in the class directory
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    # Choose random image
                    random_image = random.choice(images)
                    image_path = os.path.join(class_path, random_image)
                    
                    # Load the image
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        self.current_image_path = image_path
                        scaled_pixmap = pixmap.scaled(
                            120, 120,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self.image_display.setPixmap(scaled_pixmap)
                        self.image_loaded.emit(image_path)
                        return
                    
        # If we get here, something went wrong
        self.image_display.setText("No image")
        self.current_image_path = None
            
    def load_image(self):
        supported_formats = [f"*.{fmt.data().decode()}" for fmt in QImageReader.supportedImageFormats()]
        filter_string = "Image Files ({})".format(" ".join(supported_formats))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            filter_string
        )
        
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.current_image_path = file_path
                scaled_pixmap = pixmap.scaled(
                    120, 120,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_display.setPixmap(scaled_pixmap)
                self.image_loaded.emit(file_path)
            else:
                self.image_display.setText("Failed to load image")
                self.current_image_path = None
                
    def update_result(self, model_results):
        """Update the classification results for all models"""
        result_text = "Classification Results:\n\n"
        for model_name, result in model_results.items():
            result_text += f"{model_name}: {result}\n"
        self.results_label.setText(result_text.rstrip())