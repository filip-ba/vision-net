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
        self.setGeometry(50, 50, 1200, 804)
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
        self.export_action = QAction("Export Collections", self)
        self.import_action = QAction("Import Collections", self)
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.import_action)
        #self.export_action.triggered.connect(self.export_collections)
        #self.import_action.triggered.connect(self.import_collections)
        file_menu.addSeparator()
        # Quit action
        self.quit_action = QAction("Quit", self)
        file_menu.addAction(self.quit_action)  
        self.quit_action.triggered.connect(self.close)