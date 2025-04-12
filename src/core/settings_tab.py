from PyQt6.QtWidgets import ( QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QGroupBox, QPushButton, QLabel, QFileDialog)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, QSize, pyqtSignal
import os
import configparser


class SettingsTab(QWidget):
    style_changed = pyqtSignal(str)  
    status_message = pyqtSignal(str, int)  
    
    def __init__(self, style_manager):
        super().__init__()
        self.style_manager = style_manager
        self._create_ui()
        
    def _create_ui(self):
        """Create the settings page UI"""
        settings_layout = QVBoxLayout(self)
        
        # Dataset group
        dataset_group = QGroupBox("Dataset")
        dataset_group.setObjectName("dataset-group")
        dataset_layout = QHBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(25)

        # Load dataset button
        self.load_dataset_btn = QPushButton("Load")
        self.load_dataset_btn.setFixedWidth(150)
        self.load_dataset_btn.clicked.connect(self._load_dataset)
        
        # Current dataset path label
        self.dataset_path_label = QLabel("Current dataset: Not loaded")
        self.dataset_path_label.setWordWrap(True)
        self.dataset_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        # Load last used dataset path from config
        self._load_last_dataset_path()
        
        dataset_layout.addWidget(self.load_dataset_btn)
        dataset_layout.addWidget(self.dataset_path_label)
        settings_layout.addWidget(dataset_group)

        # Style group
        style_group = QGroupBox("Style")
        style_group.setObjectName("style-group")
        style_layout = QVBoxLayout(style_group)
        
        buttons_layout = QGridLayout()
        buttons_layout.setSpacing(20)

        project_root = self._return_project_root_folder()
        themes_dir = os.path.join(project_root, "assets/themes")
        
        # Light theme button
        self.light_button = QPushButton()
        self.light_button.setObjectName("light-style-button")
        self.light_button.setCheckable(True)
        self.light_button.setFixedSize(200, 130)
        
        light_label = QLabel("Light")
        light_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        light_label.setObjectName("style-button-label")

        light_preview_path = os.path.join(themes_dir, "light-theme-preview.png")
        if os.path.exists(light_preview_path):
            light_preview = QPixmap(light_preview_path)
            self.light_button.setIcon(QIcon(light_preview))
            self.light_button.setIconSize(QSize(180, 120))
        
        # Dark theme button
        self.dark_button = QPushButton()
        self.dark_button.setObjectName("dark-style-button")
        self.dark_button.setCheckable(True)
        self.dark_button.setFixedSize(200, 130)

        dark_label = QLabel("Dark")
        dark_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dark_label.setObjectName("style-button-label")
        
        dark_preview_path = os.path.join(themes_dir, "dark-theme-preview.png")
        if os.path.exists(dark_preview_path):
            dark_preview = QPixmap(dark_preview_path)
            self.dark_button.setIcon(QIcon(dark_preview))
            self.dark_button.setIconSize(QSize(180, 120))
        
        # Set the initially checked button based on current style
        if self.style_manager.get_current_style() == self.style_manager.STYLE_LIGHT:
            self.light_button.setChecked(True)
        else:
            self.dark_button.setChecked(True)
        
        # Add buttons and labels to the grid layout
        buttons_layout.addWidget(self.light_button, 0, 0, Qt.AlignmentFlag.AlignCenter)
        buttons_layout.addWidget(light_label, 1, 0, Qt.AlignmentFlag.AlignCenter)
        buttons_layout.addWidget(self.dark_button, 0, 1, Qt.AlignmentFlag.AlignCenter) 
        buttons_layout.addWidget(dark_label, 1, 1, Qt.AlignmentFlag.AlignCenter)
        buttons_layout.setSpacing(5)
        
        # Connect button signals
        self.light_button.clicked.connect(self._on_light_style_clicked)
        self.dark_button.clicked.connect(self._on_dark_style_clicked)
        
        style_layout.addLayout(buttons_layout)
        settings_layout.addWidget(style_group)
        
        # Add a spacer to push everything to the top
        settings_layout.addStretch()
    
    def _load_last_dataset_path(self):
        """Load the last used dataset path from config"""
        config = configparser.ConfigParser()
        config_path = "./model_config.ini"
        
        if os.path.exists(config_path):
            config.read(config_path)
            if 'Dataset' in config and 'path' in config['Dataset']:
                dataset_path = config['Dataset']['path']
                if os.path.exists(dataset_path):
                    self.dataset_path_label.setText(f"Current dataset: {dataset_path}")
                    return
        
        # If no valid path found, try default path
        default_path = "./dataset/fruitveg-dataset"
        if os.path.exists(default_path):
            self.dataset_path_label.setText(f"Current dataset: {default_path}")
        else:
            self.dataset_path_label.setText("Current dataset: Not loaded")
    
    def _save_dataset_path(self, path):
        """Save the dataset path to config file"""
        config = configparser.ConfigParser()
        config_path = "./model_config.ini"
        
        # Create or read existing config
        if os.path.exists(config_path):
            config.read(config_path)
        
        # Make sure the Dataset section exists
        if 'Dataset' not in config:
            config['Dataset'] = {}
        
        # Update the dataset path
        config['Dataset']['path'] = path
        
        # Save the config
        with open(config_path, 'w') as configfile:
            config.write(configfile)
    
    def _load_dataset(self):
        """Open file dialog to select dataset directory"""
        dataset_path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if dataset_path:
            # Check if the directory has the required structure
            required_dirs = ['train', 'valid', 'test']
            if all(os.path.exists(os.path.join(dataset_path, d)) for d in required_dirs):
                # Get the main window to access model tabs
                main_window = self.window()
                if hasattr(main_window, 'tab_widget'):
                    # Get all model tabs
                    model_tabs = []
                    for i in range(main_window.tab_widget.count()):
                        widget = main_window.tab_widget.widget(i)
                        if hasattr(widget, 'handle_dataset_change'):
                            model_tabs.append(widget)
                    
                    # If there are model tabs, show confirmation dialog
                    if model_tabs:
                        # Use the first model tab to show the dialog
                        if not model_tabs[0].handle_dataset_change():
                            return  # User cancelled
                    
                    # Update dataset path and clear models
                    self.dataset_path_label.setText(f"Current dataset: {dataset_path}")
                    self._save_dataset_path(dataset_path)
                    self.status_message.emit("Dataset loaded successfully", 8000)
            else:
                self.status_message.emit("Invalid dataset structure. Required directories: train, valid, test", 8000)
    
    def _on_light_style_clicked(self):
        if not self.light_button.isChecked():
            self.light_button.setChecked(True)
            return
            
        self.dark_button.setChecked(False)
        self.style_manager.apply_style(self.style_manager.STYLE_LIGHT)
        self.style_changed.emit("light")
        self.status_message.emit("Light theme applied", 8000)
    
    def _on_dark_style_clicked(self):
        if not self.dark_button.isChecked():
            self.dark_button.setChecked(True)
            return
            
        self.light_button.setChecked(False)
        self.style_manager.apply_style(self.style_manager.STYLE_DARK)
        self.style_changed.emit("dark")
        self.status_message.emit("Dark theme applied", 8000)
    
    def _return_project_root_folder(self):
        current_file_path = os.path.abspath(__file__)
        core_dir = os.path.dirname(current_file_path)
        src_dir = os.path.dirname(core_dir)
        project_root = os.path.dirname(src_dir)
        return project_root 