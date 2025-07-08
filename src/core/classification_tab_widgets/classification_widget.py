from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy)
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QSize

import os


class ClassificationWidget(QWidget):
    image_loaded = pyqtSignal(str, int)
    load_img_btn_clicked = pyqtSignal()
    classify_img_btn_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = None
        self.original_pixmap = None
        self._create_ui()

    def set_arrow_icons(self, icons_dir, theme_suffix):
        next_btn_icon_path = os.path.join(icons_dir, f"next-{theme_suffix}.png")
        self.next_btn.setIcon(QIcon(next_btn_icon_path))
        self.next_btn.setIconSize(QSize(16, 16))
        previous_btn_icon_path = os.path.join(icons_dir, f"previous-{theme_suffix}.png")
        self.prev_btn.setIcon(QIcon(previous_btn_icon_path))
        self.prev_btn.setIconSize(QSize(16, 16))

    def update_navigation_buttons(self, prev_enabled, next_enabled):
        self.prev_btn.setEnabled(prev_enabled)
        self.next_btn.setEnabled(next_enabled)
        
    def update_image(self, image_path):
        if not image_path:
            return
            
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
            
        self.original_pixmap = pixmap
        self.current_image_path = image_path
        self.scale_image()
        
    def scale_image(self):
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
            
        display_size = self.image_display.size()
        scaled_pixmap = self.original_pixmap.scaled(
            display_size.width(), 
            display_size.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_display.setPixmap(scaled_pixmap)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scale_image() 

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        group_box = QGroupBox("Image Classification")
        group_box.setObjectName("classification-group")
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(15, 15, 15, 15)
        group_layout.setSpacing(10)
        
        self.image_display = QLabel()
        self.image_display.setObjectName("image-display")
        self.image_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_display.setMinimumSize(100, 100)  
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.prev_btn = QPushButton()
        self.prev_btn.setToolTip("Previous test image")
        self.prev_btn.setObjectName("prev-btn")
        self.next_btn = QPushButton()
        self.next_btn.setToolTip("Next test image")
        self.next_btn.setObjectName("next-btn")
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        top_part_layout = QVBoxLayout()
        top_part_layout.setContentsMargins(0, 0, 0, 0)
        top_part_layout.setSpacing(0)
        top_part_layout.addWidget(self.image_display)
        top_part_layout.addLayout(nav_layout)

        group_layout.addLayout(top_part_layout)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        self.load_btn = QPushButton("Load")
        self.classify_btn = QPushButton("Classify")
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.classify_btn)
        
        group_layout.addLayout(button_layout)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        self.setLayout(layout)
        
        self.load_btn.clicked.connect(self.load_img_btn_clicked.emit) 
        self.classify_btn.clicked.connect(self.classify_img_btn_clicked.emit)
        self.prev_btn.clicked.connect(self.prev_clicked.emit)
        self.next_btn.clicked.connect(self.next_clicked.emit)