from PyQt6.QtWidgets import (QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                             QGroupBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
import configparser
import sys
import os

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
        config_path = os.path.join(project_root, "model_config.ini")

        if not os.path.exists(config_path):
            self.status_message.emit("Config not found: model_config.ini", 8000)
            return None

        config.read(config_path)
        if config.has_option('DatasetPath', 'dataset_path'):
            dataset_path = config.get('DatasetPath', 'dataset_path')
            if os.path.exists(dataset_path):
                return dataset_path
            else:
                self.status_message.emit(f"Saved dataset path not found: {dataset_path}", 8000)
                return None
        else:
            self.status_message.emit("No path saved in the config file.", 8000)
            return None

    def _save_dataset_path(self, dataset_path):
        project_root = self.get_project_root()
        config_path = os.path.join(project_root, "model_config.ini")
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

        dataset_overview_group = QGroupBox("Dataset Overview")
        dataset_overview_group.setObjectName("dataset-overview")
        
        dataset_overview_group_layout = QHBoxLayout()
        dataset_overview_group_layout.setContentsMargins(0, 0, 0, 0)
        dataset_overview_group_layout.setSpacing(20)
        
        self.load_dataset_btn = QPushButton("Load Dataset")

        self.dataset_path_label = QLabel("Dataset path will be displayed here.")
        self.dataset_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.dataset_path_label.setStyleSheet("color: gray; font-style: italic;")

        dataset_overview_group_layout.addWidget(self.load_dataset_btn, 1)
        dataset_overview_group_layout.addWidget(self.dataset_path_label, 4)
        dataset_overview_group.setLayout(dataset_overview_group_layout)

        self.dataset_status_widget = DatasetStatusWidget(self.model_tabs)

        main_layout.addWidget(dataset_overview_group)
        main_layout.addWidget(self.dataset_status_widget)
        main_layout.addStretch()