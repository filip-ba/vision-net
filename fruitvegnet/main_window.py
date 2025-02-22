from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
    QFileDialog, QLabel, QPushButton, QHBoxLayout, QGridLayout
)
from PyQt6.QtGui import QIcon, QPixmap, QImageReader, QAction
from PyQt6.QtCore import Qt, pyqtSignal
import os
import random

from fruitvegnet.main_widget import MainWidget
from fruitvegnet.image_classification_widget import ImageClassificationWidget
from models.simple_cnn_model import SimpleCnnModel
from models.resnet_model import ResNetModel
from models.efficientnet_model import EfficientNetModel
from models.vgg16_model import VGG16Model


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._create_ui()
        
    def _create_ui(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(project_root, "assets", "icon.ico")  
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("FruitVegNet")
        self.setGeometry(50, 50, 1200, 920)
        
        # Main widget setup
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Shared image widget
        self.shared_image = ImageClassificationWidget()
        main_layout.addWidget(self.shared_image)
        
        # Create tab widget
        self.tab_widget = QTabWidget(self)
        self.simple_cnn_tab = MainWidget(model_class=SimpleCnnModel)
        self.resnet_tab = MainWidget(model_class=ResNetModel)
        self.efficientnet_tab = MainWidget(model_class=EfficientNetModel)
        self.vgg16_tab = MainWidget(model_class=VGG16Model)
        
        # Hide individual image controls in tabs
        for tab in [self.simple_cnn_tab, self.resnet_tab, self.efficientnet_tab, self.vgg16_tab]:
            tab.image_widget.load_image_btn.hide()
            tab.image_widget.image_display.hide()
        
        self.tab_widget.addTab(self.simple_cnn_tab, "Simple CNN")
        self.tab_widget.addTab(self.resnet_tab, "ResNet")
        self.tab_widget.addTab(self.efficientnet_tab, "EfficientNet-B0")
        self.tab_widget.addTab(self.vgg16_tab, "VGG16")
        main_layout.addWidget(self.tab_widget)
        
        # Connect shared image signals
        self.shared_image.image_loaded.connect(self._update_all_tabs_image)
        self.shared_image.classify_clicked.connect(self._classify_all)
        
        # MenuBar setup
        self._setup_menu()
        
    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        # Export and import actions
        self.save_action = QAction("Save Model", self)
        self.load_action = QAction("Load Model", self)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.load_action)
        self.save_action.triggered.connect(self._save_current_model)
        self.load_action.triggered.connect(self._load_current_model)
        file_menu.addSeparator()
        
        # Quit action
        self.quit_action = QAction("Quit", self)
        file_menu.addAction(self.quit_action)  
        self.quit_action.triggered.connect(self.close)
        
    def _update_all_tabs_image(self, image_path):
        """Update the image in all tabs when a new image is loaded"""
        tabs = [self.simple_cnn_tab, self.resnet_tab, self.efficientnet_tab, self.vgg16_tab]
        for tab in tabs:
            tab.current_image_path = image_path
            tab.image_widget.result_label.setText("Classification:\nNone")
            tab.image_widget.init_plot()
            
    def _classify_all(self):
        """Classify the current image using all loaded models"""
        if not self.shared_image.current_image_path:
            return
            
        model_map = {
            self.simple_cnn_tab: 'Simple CNN',
            self.resnet_tab: 'ResNet',
            self.efficientnet_tab: 'EfficientNet',
            self.vgg16_tab: 'VGG16'
        }
        
        results = {}
        for tab, model_name in model_map.items():
            if tab.model_loaded:
                try:
                    result = tab.model.predict_image(self.shared_image.current_image_path)
                    predicted_class = result['class']
                    probabilities = result['probabilities']
                    
                    # Update tab's visualization
                    tab.image_widget.result_label.setText(f"Classification:\n{predicted_class}")
                    tab.image_widget.update_plot(tab.model.classes, probabilities)
                    
                    # Store result
                    results[model_name] = predicted_class
                except Exception as e:
                    results[model_name] = "Error"
            else:
                results[model_name] = "No model"
                
        # Update the shared widget's results
        self.shared_image.update_result(results)
                
    def _save_current_model(self):
        """Save the model from the currently active tab"""
        current_tab = self.tab_widget.currentWidget()
        if current_tab:
            current_tab.save_model()
            
    def _load_current_model(self):
        """Load a model into the currently active tab"""
        current_tab = self.tab_widget.currentWidget()
        if current_tab:
            current_tab.load_model()