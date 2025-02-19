from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtGui import QAction, QIcon
import os
from fruitvegnet.main_widget import MainWidget
from models.simple_cnn_model import SimpleCnnModel
from models.resnet_model import ResNetModel
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
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        # Create tab widget
        self.tab_widget = QTabWidget(self)
        # Create tabs with different models
        self.simple_cnn_tab = MainWidget(model_class=SimpleCnnModel)
        self.resnet_tab = MainWidget(model_class=ResNetModel)
        self.vgg16_tab = MainWidget(model_class=VGG16Model)
        self.tab_widget.addTab(self.simple_cnn_tab, "Simple CNN Model")
        self.tab_widget.addTab(self.resnet_tab, "ResNet Model")
        self.tab_widget.addTab(self.vgg16_tab, "VGG16 Model")
        main_layout.addWidget(self.tab_widget)
        # MenuBar
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