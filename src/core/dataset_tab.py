from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtCore import pyqtSignal, QTimer
import configparser
import sys
import os

from .dataset_tab_widgets.dataset_overview_widget import DatasetOverviewWidget
from .dataset_tab_widgets.dataset_status_widget import DatasetStatusWidget


class DatasetTab(QWidget):
    status_message = pyqtSignal(str, int)

    def __init__(self, model_tabs, parent=None):
        super().__init__(parent)

        self.model_tabs = model_tabs

        self._create_ui()

        self.load_dataset_btn.clicked.connect(self._browse_dataset)

        QTimer.singleShot(0, self._load_dataset_on_start)

    def _browse_dataset(self):
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
            self.dataset_path_label.setText(dir_path)
            self._load_dataset(dir_path)
            self._save_dataset_path(dir_path)

    def _load_dataset_on_start(self):
        dataset_path = self._load_dataset_path_from_config()
        if not dataset_path == None:
            self.dataset_path_label.setText(dataset_path)
            self._load_dataset(dataset_path)
            
    def _load_dataset(self, dataset_path):
        try:
            simple_cnn_tab = self.model_tabs[0]
            simple_cnn_tab.model.load_dataset(dataset_path)
            self.dataset_status_widget.set_status(simple_cnn_tab.model_name, "OK", "green")
        except Exception as e:
            self.dataset_status_widget.set_status(simple_cnn_tab.model_name, str(e), "red")

        pretrained_tabs = self.model_tabs[1:]
        source_tab = None  

        for tab in pretrained_tabs:
            if source_tab is None:
                try:
                    tab.model.load_dataset(dataset_path)
                    self.dataset_status_widget.set_status(tab.model_name, "OK", "green")
                    source_tab = tab 
                except Exception as e:
                    self.dataset_status_widget.set_status(tab.model_name, str(e), "red")
            else:
                try:
                    tab.model.share_dataset(source_tab.model)
                    self.dataset_status_widget.set_status(tab.model_name, "OK", "green")
                except Exception as e:
                    self.dataset_status_widget.set_status(tab.model_name, str(e), "red")  

    def _load_dataset_path_from_config(self):
        project_root = self.get_project_root()
        config = configparser.ConfigParser()
        config_path = os.path.join(project_root, "config.ini")

        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                config.write(f)

        config.read(config_path)
        if config.has_option('DatasetPath', 'dataset_path'):
            dataset_path = config.get('DatasetPath', 'dataset_path')
            if os.path.exists(dataset_path):
                return dataset_path
            else:
                self.status_message.emit(f"Saved dataset path not found: {dataset_path}", 12000)
                return None
        else:
            self.status_message.emit("Dataset not loaded.", 10000)
            return None

    def _save_dataset_path(self, dataset_path):
        project_root = self.get_project_root()
        config_path = os.path.join(project_root, "config.ini")
        config = configparser.ConfigParser()
        config.read(config_path)

        if not config.has_section('DatasetPath'):
            config.add_section('DatasetPath')

        config.set('DatasetPath', 'dataset_path', dataset_path)

        with open(config_path, 'w') as configfile:
            config.write(configfile)
        self.status_message.emit(f"Dataset path saved: {dataset_path}", 4000)

    def get_project_root(self):
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            current_file_path = os.path.abspath(__file__)
            core_dir = os.path.dirname(current_file_path)
            src_dir = os.path.dirname(core_dir)
            project_root = os.path.dirname(src_dir)
            return project_root

    def _create_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)

        self.dataset_overview_widget = DatasetOverviewWidget(self)
        self.load_dataset_btn = self.dataset_overview_widget.load_dataset_btn
        self.dataset_path_label = self.dataset_overview_widget.dataset_path_label

        self.dataset_status_widget = DatasetStatusWidget(self, self.model_tabs)

        main_layout.addWidget(self.dataset_overview_widget)
        main_layout.addWidget(self.dataset_status_widget)
        main_layout.addStretch()