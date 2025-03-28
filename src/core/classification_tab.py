from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog 
from PyQt6.QtGui import QPixmap, QImageReader
from PyQt6.QtCore import pyqtSignal, QTimer
import os, random

from src.ui.widgets.image_classification_widget import ImageClassificationWidget
from src.ui.widgets.results_widget import ResultsWidget
from src.ui.widgets.plot_probability_widget import PlotProbabilityWidget


class ClassificationTab(QWidget):
    image_loaded = pyqtSignal(str, int)  
    classify_clicked = pyqtSignal()  

    def __init__(self):
        super().__init__()
        self.current_image_path = None
        
        self.model_names = {
            'simple_cnn': 'Simple CNN',
            'resnet': 'ResNet',
            'efficientnet': 'EfficientNet',
            'vgg16': 'VGG16'
        }
        
        self._create_ui()    
        
        # Load random image on startup
        self.load_random_test_image()
        
    def load_random_test_image(self):
        dataset_path = "./dataset/fruitveg-dataset/test"

        if os.path.exists(dataset_path):
            class_dirs = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
            
            if class_dirs:
                random_class = random.choice(class_dirs)
                class_path = os.path.join(dataset_path, random_class)
                
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    random_image = random.choice(images)
                    image_path = os.path.join(class_path, random_image)
                    
                    self.current_image_path = image_path
                    self.image_widget.update_image_display(image_path)
                    return
                    
        # If no image could be loaded
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
                self.image_widget.update_image_display(file_path)
                self.image_loaded.emit("Image loaded", 8000)
            else:
                self.current_image_path = None
                self.image_loaded.emit("Failed to load image", 8000)
                
    def update_result(self, model_type, result):
        """Update the classification result for a specific model"""
        self.results_widget.update_result(model_type, result)

    def update_plot(self, model_type, classes, probabilities):
        """Update probability plot for a specific model"""
        self.plot_widget.update_plot(model_type, classes, probabilities)
        
    def init_plot(self, model_type=None):
        """Initialize empty probability plot"""
        self.plot_widget.init_plot(model_type)
        
    def switch_plot(self, model_type):
        """Switch to the specified plot"""
        self.plot_widget.switch_plot(model_type)

    def showEvent(self, event):
        """Scale the image in image display properly after the start of the application"""
        super().showEvent(event)
        # Scale the image after the widget is visible and has proper dimensions
        QTimer.singleShot(50, self.image_widget.scale_image)

    def _create_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 0)
        main_layout.setSpacing(0)

        # Create top and bottom layouts
        top_layout = self._create_top_layout()
        bottom_layout = self._create_bottom_layout()
        
        main_layout.addLayout(top_layout, 4)
        main_layout.addLayout(bottom_layout, 6)
        
        self.setLayout(main_layout)
        
        # Connect signals
        self.image_widget.load_btn.clicked.connect(self.load_image)
        self.image_widget.classify_clicked.connect(self.classify_clicked.emit)
        
    def _create_top_layout(self):
        # Top section for classification and results group boxes 
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # Create image classification widget
        self.image_widget = ImageClassificationWidget()
        
        # Create results widget
        self.results_widget = ResultsWidget()
        
        top_layout.addWidget(self.image_widget, 2)
        top_layout.addWidget(self.results_widget, 3)
        
        return top_layout
        
    def _create_bottom_layout(self):
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        
        # Create probability plots widget
        self.plot_widget = PlotProbabilityWidget()
        bottom_layout.addWidget(self.plot_widget)
        
        return bottom_layout