from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtCore import pyqtSignal, QTimer
import configparser
import os

from .dataset_tab_widgets.dataset_overview_widget import DatasetOverviewWidget
from .dataset_tab_widgets.dataset_status_widget import DatasetStatusWidget
from ..utils.get_project_root import get_project_root


class DatasetTab(QWidget):
    status_message = pyqtSignal(str, int)
    dataset_loaded = pyqtSignal(str)  

    def __init__(self, model_tabs, parent=None):
        super().__init__(parent)
        self.model_tabs = model_tabs
        self.dataset_path = None
        self._create_ui()

        self.load_dataset_btn.clicked.connect(self._browse_dataset)
        self.dataset_status_widget.check_status_btn.clicked.connect(self.check_dataset_status)
        QTimer.singleShot(0, self._load_dataset_on_start)

    def _browse_dataset(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Load dataset")
        msg_box.setText("This will clear all loaded models. Are you sure you want to load a different dataset?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        msg_box.setObjectName("clear-model-message")
        reply = msg_box.exec()
        if reply == QMessageBox.StandardButton.No:
            return

        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            required_folders = ["train", "test", "valid"]
            subdirs = set(os.listdir(dir_path))
            missing = [fld for fld in required_folders if fld not in subdirs]
            if missing:
                QMessageBox.critical(self, "Invalid Dataset Structure", 
                    "The selected directory must contain 'train', 'test' and 'valid' subfolders.")
                return
            self.status_message.emit(f"Selected dataset: {dir_path}", 8000)
            self._save_dataset_path_to_config(dir_path)
            self._load_dataset(dir_path)

    def _load_dataset_on_start(self):
        dataset_path = self._load_dataset_path_from_config()
        if not dataset_path == None:
            self._load_dataset(dataset_path)
            for tab in self.model_tabs:
                tab.load_model_on_start()
            
    def _load_dataset(self, dataset_path):
        self.dataset_path_label.setText(dataset_path)
        self.dataset_path = dataset_path
        success = False
        try:
            simple_cnn_tab = self.model_tabs[0]
            simple_cnn_tab.reset_model()
            simple_cnn_tab.model.load_dataset(dataset_path)
            self.dataset_status_widget.set_status(simple_cnn_tab.model_name, "OK", "green")
            success = True
        except Exception as e:
            self.dataset_status_widget.set_status(simple_cnn_tab.model_name, str(e), "red")

        pretrained_tabs = self.model_tabs[1:]
        source_tab = None  

        for tab in pretrained_tabs:
            tab.reset_model()
            if source_tab is None:
                try:
                    tab.model.load_dataset(dataset_path)
                    self.dataset_status_widget.set_status(tab.model_name, "OK", "green")
                    source_tab = tab
                    success = True 
                except Exception as e:
                    self.dataset_status_widget.set_status(tab.model_name, str(e), "red")
            else:
                try:
                    tab.model.share_dataset(source_tab.model)
                    self.dataset_status_widget.set_status(tab.model_name, "OK", "green")
                    success = True
                except Exception as e:
                    self.dataset_status_widget.set_status(tab.model_name, str(e), "red")  

        if success:
            self.status_message.emit(f"Dataset loaded: {os.path.basename(dataset_path)}", 8000)
            self.dataset_loaded.emit(dataset_path)  # Signal that connects to self.classification_tab.load_test_images in main_window

    def _load_dataset_path_from_config(self):
        project_root = get_project_root()
        config = configparser.ConfigParser()
        config_path = os.path.join(project_root, "config.ini")

        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                config.write(f)

        config.read(config_path)

        if config.has_option('DatasetPath', 'dataset_path'):
            dataset_path = config.get('DatasetPath', 'dataset_path')
            if not os.path.exists(dataset_path):
                self.status_message.emit(f"Saved dataset path not found: {dataset_path}", 12000)
            return dataset_path
        else:
            self.status_message.emit("Dataset not loaded.", 10000)
            return None

    def _save_dataset_path_to_config(self, dataset_path):
        project_root = get_project_root()
        config_path = os.path.join(project_root, "config.ini")
        config = configparser.ConfigParser()
        config.read(config_path)

        if not config.has_section('DatasetPath'):
            config.add_section('DatasetPath')

        config.set('DatasetPath', 'dataset_path', dataset_path)

        with open(config_path, 'w') as configfile:
            config.write(configfile)
    
    def check_dataset_status(self):
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            self.status_message.emit("Dataset path is not set or not found.", 8000)
            for tab in self.model_tabs:
                self.dataset_status_widget.set_status(tab.model_name, "Not Found", "red")
                tab.set_button_state(False, False)

        self.status_message.emit("Checking dataset status...", 2000)

    def _create_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.dataset_overview_widget = DatasetOverviewWidget(self)
        self.load_dataset_btn = self.dataset_overview_widget.load_dataset_btn
        self.dataset_path_label = self.dataset_overview_widget.dataset_path_label

        self.dataset_status_widget = DatasetStatusWidget(self, self.model_tabs)

        main_layout.addWidget(self.dataset_overview_widget)
        main_layout.addWidget(self.dataset_status_widget)
        main_layout.addStretch()