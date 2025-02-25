from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, 
    QFileDialog, QLabel, QPushButton, QStackedWidget
)
from PyQt6.QtGui import QPixmap, QImageReader, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os, random
from matplotlib import pyplot as plt


class ImageClassificationWidget(QWidget):
    image_loaded = pyqtSignal(str)  # Signal to notify when new image is loaded
    classify_clicked = pyqtSignal()  # Signal for classify button clicks
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self._create_ui()
        
    def _create_ui(self):
        # Main group box
        main_group = QGroupBox("Image Classification")
        main_layout = QHBoxLayout(main_group)
        
        # Left side - Image and buttons
        left_layout = QVBoxLayout()
        
        # Image preview
        self.image_display = QLabel()
        self.image_display.setFixedSize(150, 150)
        self.image_display.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.image_display)
        
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
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
            """)
            button_layout.addWidget(btn)
        left_layout.addLayout(button_layout)
        left_layout.addStretch()
        main_layout.addLayout(left_layout)
        
        # Middle - Result labels with Show Plot buttons
        middle_layout = QVBoxLayout()
        self.result_labels = {}
        self.plot_buttons = {}
        
        for model_type, label_text in {
            'simple_cnn': 'Simple CNN',
            'resnet': 'ResNet',
            'efficientnet': 'EfficientNet',
            'vgg16': 'VGG16'
        }.items():
            # Create container for each model
            container = QWidget()
            container_layout = QHBoxLayout(container)
            
            # Create and style label
            label = QLabel(f"{label_text}: None")
            label.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    color: #2c3e50;
                    padding: 8px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    text-align: left;
                }
            """)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFixedSize(200, 40)
            container_layout.addWidget(label)
            

            # Create plot button
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            icon_path = os.path.join(project_root, "assets", "graph_icon.png")
            #icon_path = "./assets/graph_icon.png"

            # Ověření načtení obrázku
            pixmap = QPixmap(icon_path)
            if pixmap.isNull():
                print("Chyba: Obrázek se nenačetl!")

            # Vytvoření tlačítka s ikonou
            plot_btn = QPushButton()
            icon = QIcon(pixmap)  # Použití pixmapy pro vytvoření ikony
            plot_btn.setIcon(icon)
            plot_btn.setIconSize(QSize(32, 32))  # Nastavení velikosti ikony
            plot_btn.setFixedSize(40, 40)

            container_layout.addWidget(plot_btn)
            
            # Store references
            self.result_labels[model_type] = label
            self.plot_buttons[model_type] = plot_btn
            
            middle_layout.addWidget(container)
        
        middle_layout.addStretch()
        main_layout.addLayout(middle_layout)
        
        # Right side - Stacked Plot Widget
        self.plot_stack = QStackedWidget()
        
        # Create plot widgets for each model
        self.plot_widgets = {}
        for model_type in ['simple_cnn', 'resnet', 'efficientnet', 'vgg16']:
            figure = Figure(figsize=(5, 4))
            canvas = FigureCanvas(figure)
            self.plot_widgets[model_type] = {'figure': figure, 'canvas': canvas}
            self.plot_stack.addWidget(canvas)
            
            # Initialize empty plot
            self.init_plot(model_type)
            
            # Connect button to show this plot
            self.plot_buttons[model_type].clicked.connect(
                lambda checked, m=model_type: self.plot_stack.setCurrentWidget(
                    self.plot_widgets[m]['canvas']
                )
            )
        
        main_layout.addWidget(self.plot_stack)
        
        # Set up main widget layout
        widget_layout = QVBoxLayout(self)
        widget_layout.addWidget(main_group)
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_image)
        self.classify_btn.clicked.connect(self.classify_clicked.emit)
        
        # Load initial image from dataset
        self.load_random_test_image()
        
    def init_plot(self, model_type=None):
        """Initialize empty probability plot for specified or all models"""
        if model_type:
            models = [model_type]
        else:
            models = self.plot_widgets.keys()
            
        for model in models:
            figure = self.plot_widgets[model]['figure']
            figure.clear()
            ax = figure.add_subplot(111)
            ax.set_title('Class Probabilities', fontsize=10)
            ax.set_xlabel('Class', fontsize=8)
            ax.set_ylabel('Probability', fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_ylim(0, 1)
            figure.tight_layout()
            self.plot_widgets[model]['canvas'].draw()

    def update_plot(self, model_type, classes, probabilities):
        """Update probability plot for a specific model"""
        if model_type not in self.plot_widgets:
            return
            
        figure = self.plot_widgets[model_type]['figure']
        figure.clear()
        ax = figure.add_subplot(111)
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
        figure.tight_layout()
        self.plot_widgets[model_type]['canvas'].draw()

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
                            150, 150,
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
                    150, 150,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_display.setPixmap(scaled_pixmap)
                self.image_loaded.emit(file_path)
            else:
                self.image_display.setText("Failed to load image")
                self.current_image_path = None
                
    def update_result(self, model_type, result):
        """Update the classification result for a specific model"""
        if model_type in self.result_labels:
            model_name = {
                'simple_cnn': 'Simple CNN',
                'resnet': 'ResNet',
                'efficientnet': 'EfficientNet',
                'vgg16': 'VGG16'
            }[model_type]
            self.result_labels[model_type].setText(f"{model_name}:{result}")