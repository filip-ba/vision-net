from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QSizePolicy
from PyQt6.QtGui import QIcon, QAction
import os

from fruitvegnet.tab_widget import TabWidget
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
        
        # Image classification widget
        self.image_classification_widget = ImageClassificationWidget()
        self.image_classification_widget.setMaximumHeight(300)
        main_layout.addWidget(self.image_classification_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Roztažení na zbytek prostoru
        self.simple_cnn_tab = TabWidget(model_class=SimpleCnnModel)
        self.resnet_tab = TabWidget(model_class=ResNetModel)
        self.efficientnet_tab = TabWidget(model_class=EfficientNetModel)
        self.vgg16_tab = TabWidget(model_class=VGG16Model)
        
        self.tab_widget.addTab(self.simple_cnn_tab, "Simple CNN")
        self.tab_widget.addTab(self.resnet_tab, "ResNet")
        self.tab_widget.addTab(self.efficientnet_tab, "EfficientNet-B0")
        self.tab_widget.addTab(self.vgg16_tab, "VGG16")
        main_layout.addWidget(self.tab_widget)
        
        # Connect shared image signals
        self.image_classification_widget.image_loaded.connect(self._update_all_tabs_image)
        self.image_classification_widget.classify_clicked.connect(self._classify_all)
        
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
        if not self.image_classification_widget.current_image_path:
            return
            
        model_map = {
            self.simple_cnn_tab: {'type': 'simple_cnn', 'name': 'Simple CNN'},
            self.resnet_tab: {'type': 'resnet', 'name': 'ResNet'},
            self.efficientnet_tab: {'type': 'efficientnet', 'name': 'EfficientNet'},
            self.vgg16_tab: {'type': 'vgg16', 'name': 'VGG16'}
        }
        
        for tab, model_info in model_map.items():
            if tab.model_loaded:
                try:
                    # Get prediction
                    result = tab.model.predict_image(self.image_classification_widget.current_image_path)
                    predicted_class = result['class']
                    probabilities = result['probabilities']
                    
                    # Update result label
                    self.image_classification_widget.update_result(model_info['type'], predicted_class)
                    
                    # Update plot for this model
                    self.image_classification_widget.update_plot(model_info['type'], tab.model.classes, probabilities)
                    
                except Exception as e:
                    self.image_classification_widget.update_result(model_info['type'], "Error")
                    print(f"Error in {model_info['name']} classification: {str(e)}")
            else:
                self.image_classification_widget.update_result(model_info['type'], "No model")

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