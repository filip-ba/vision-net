from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import pyqtSignal, QObject
import os
from gui.pytorch_tab import PytorchTab
from gui.keras_tab import KerasTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._create_ui()
        
    def _create_ui(self):
        project_dir = os.path.dirname(os.path.dirname(__file__))
        icon_path = os.path.join(project_dir, "icon.ico")
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("FruitVegNet")
        self.setGeometry(50, 50, 1200, 900)
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        self.tab_widget = QTabWidget(self)
        # PyTorch and Keras Tab
        pytorch_tab = PytorchTab()
        keras_tab = KerasTab()
        self.tab_widget.addTab(pytorch_tab, "PyTorch")
        self.tab_widget.addTab(keras_tab, "Keras")
        main_layout.addWidget(self.tab_widget)
        # MenuBar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        # Export and import actions
        self.save_action = QAction("Save Model", self)
        self.load_action = QAction("Load Model", self)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.load_action)
        self.save_action.triggered.connect(pytorch_tab.save_model)
        self.load_action.triggered.connect(pytorch_tab.load_model)
        file_menu.addSeparator()
        # Quit action
        self.quit_action = QAction("Quit", self)
        file_menu.addAction(self.quit_action)  
        self.quit_action.triggered.connect(self.close)