from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QGroupBox, QPushButton, QLabel
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, QSize, pyqtSignal
import os


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
        
        style_group = QGroupBox("Style")
        style_group.setObjectName("style-group")
        style_layout = QVBoxLayout(style_group)
        
        buttons_layout = QGridLayout()
        buttons_layout.setSpacing(20)

        project_root = self._return_project_root_folder()
        themes_dir = os.path.join(project_root, "assets", "themes")
        
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
        
        # Connect button signals
        self.light_button.clicked.connect(self._on_light_style_clicked)
        self.dark_button.clicked.connect(self._on_dark_style_clicked)
        
        style_layout.addLayout(buttons_layout)
        settings_layout.addWidget(style_group)
        
        # Add a spacer to push everything to the top
        settings_layout.addStretch()
    
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