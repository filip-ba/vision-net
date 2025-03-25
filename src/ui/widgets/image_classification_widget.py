from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal


class ImageClassificationWidget(QWidget):
    image_loaded = pyqtSignal(str, int)
    classify_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = None
        self.original_pixmap = None
        self._create_ui()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        group_box = QGroupBox("Image Classification")
        group_box.setObjectName("classification-group")
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(15, 15, 15, 15)
        group_layout.setSpacing(15)
        
        self.image_display = QLabel()
        self.image_display.setObjectName("image-display")
        self.image_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_display.setMinimumSize(100, 100)  
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_layout.addWidget(self.image_display)
        
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        self.load_btn = QPushButton("Load")
        self.classify_btn = QPushButton("Classify")
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.classify_btn)
        
        group_layout.addLayout(button_layout)
        group_box.setLayout(group_layout)
        
        layout.addWidget(group_box)
        self.setLayout(layout)
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_image)
        self.classify_btn.clicked.connect(self.classify_clicked.emit)
        
    def load_image(self):
        """Signal to parent to load an image"""
        pass
        
    def update_image_display(self, image_path):
        """Update the image display with the specified image"""
        if not image_path:
            return
            
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
            
        self.original_pixmap = pixmap
        self.current_image_path = image_path
        self.scale_image()
        
    def scale_image(self):
        """Scale image to fit display while preserving aspect ratio"""
        if not hasattr(self, 'original_pixmap') or self.original_pixmap.isNull():
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
        """Scale the image in image display on window resize event"""
        super().resizeEvent(event)
        self.scale_image() 