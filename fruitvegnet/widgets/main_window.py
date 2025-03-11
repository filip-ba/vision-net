from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QSizePolicy, QSpacerItem, QStatusBar, QFrame,
                             QHBoxLayout, QStackedWidget, QPushButton, QLabel,
                             QScrollArea)
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtCore import Qt, QSize, QTimer, QEvent
import os

from fruitvegnet.model_settings import TabWidget
from fruitvegnet.image_classification import ImageClassification
from models.simple_cnn_model import SimpleCnnModel
from models.resnet_model import ResNetModel
from models.efficientnet_model import EfficientNetModel
from models.vgg16_model import VGG16Model


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        # Add responsive sidebar settings
        self.MIN_SIDEBAR_WIDTH = 190
        self.MAX_SIDEBAR_WIDTH = 260
        self.SIDEBAR_HIDE_THRESHOLD = 500  # Hide sidebar when window width is less than this
        
        # Install event filter for window resize
        self.installEventFilter(self)
        
        self._create_ui()
        
    def _create_ui(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(project_root, "assets", "icon.png")  

        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("FruitVegNet")
        self.setGeometry(50, 50, 1100, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create scrollable sidebar
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.sidebar = self._create_sidebar()
        self.sidebar_scroll.setWidget(self.sidebar)
        
        # Set fixed minimum width to start
        self.sidebar_scroll.setMinimumWidth(self.MIN_SIDEBAR_WIDTH)
        self.sidebar_scroll.setMaximumWidth(self.MIN_SIDEBAR_WIDTH)
        
        # Container for the model settings and image classification
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Create a scrollable area for the content stack
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create stacked widget for switching between settings and classification
        self.content_stack = QStackedWidget()
        
        # Create model settings (first page - tab widget)
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  
        self.simple_cnn_tab = TabWidget(model_class=SimpleCnnModel)
        self.resnet_tab = TabWidget(model_class=ResNetModel)
        self.efficientnet_tab = TabWidget(model_class=EfficientNetModel)
        self.vgg16_tab = TabWidget(model_class=VGG16Model)
        
        self.tab_widget.addTab(self.simple_cnn_tab, "Simple CNN")
        self.tab_widget.addTab(self.resnet_tab, "ResNet")
        self.tab_widget.addTab(self.efficientnet_tab, "EfficientNet-B0")
        self.tab_widget.addTab(self.vgg16_tab, "VGG16")
        
        # Create image classification (second page)
        self.image_classification_widget = ImageClassification()
        self.image_classification_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create settings page (third page)
        self.settings_widget = QWidget() 
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_label = QLabel("Settings Page")
        settings_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_layout.addWidget(settings_label)
        self.settings_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.content_stack.addWidget(self.tab_widget)
        self.content_stack.addWidget(self.image_classification_widget)
        self.content_stack.addWidget(self.settings_widget) 

        # Add the stacked widget to the scroll area
        content_scroll.setWidget(self.content_stack)

        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("status-bar")
        
        content_layout.addWidget(content_scroll, 1)
        content_layout.addWidget(self.status_bar, 0)
        
        # Add widgets to main layout
        self.main_layout.addWidget(self.sidebar_scroll, 1)  
        self.main_layout.addWidget(content_container, 5)  
        
        # Signals
        self.image_classification_widget.classify_clicked.connect(self._classify_all) 
        self.image_classification_widget.image_loaded.connect(self.update_status_bar) 
        self._connect_tab_status_signals()

    def _create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Logo and title
        logo_widget = QWidget()
        logo_widget.setObjectName("logo-widget")
        logo_widget.setMinimumHeight(80)
        logo_layout = QVBoxLayout(logo_widget)
        
        title_label = QLabel("FruitVegNet")
        title_label.setObjectName("app-title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(title_label)
        
        sidebar_layout.addWidget(logo_widget)

        # Add custom separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setStyleSheet("color: #e3e3e3; background-color: #f2f2f2; height: 1px; margin-left: 10px; margin-right: 10px;")

        sidebar_layout.addWidget(separator)

        # Add empty space under the separator 
        spacer = QSpacerItem(20, 30, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sidebar_layout.addItem(spacer)
        
        # Add navigation buttons
        models_btn = self._create_sidebar_button("Models", 0)
        classify_btn = self._create_sidebar_button("Classification", 1)
        
        sidebar_layout.addWidget(models_btn)
        sidebar_layout.addWidget(classify_btn)
        
        # Add stretch to push the settings button to the bottom
        sidebar_layout.addStretch()
        
        # Add separator above settings button
        bottom_separator = QFrame()
        bottom_separator.setFrameShape(QFrame.Shape.HLine)
        bottom_separator.setFrameShadow(QFrame.Shadow.Plain)
        bottom_separator.setStyleSheet("color: #e3e3e3; background-color: #f2f2f2; height: 1px; margin-left: 10px; margin-right: 10px;")
        sidebar_layout.addWidget(bottom_separator)
        
        # Add Settings button at the bottom
        settings_btn = self._create_sidebar_button("Settings", 2)
        sidebar_layout.addWidget(settings_btn)
        
        return sidebar
    
    def _create_sidebar_button(self, text, page_index):
        button = QPushButton(text)
        button.setObjectName("sidebar-button")
        button.setCheckable(True)
        
        # Path to assets folder
        current_file_path = os.path.abspath(__file__)
        widgets_dir = os.path.dirname(current_file_path)
        fruitvegnet_dir = os.path.dirname(widgets_dir)
        project_root = os.path.dirname(fruitvegnet_dir)
        assets_dir = os.path.join(project_root, "assets")
        
        # Set icons
        icon_path = None
        if text == "Models":
            icon_path = os.path.join(assets_dir, "model-dark.png")
        elif text == "Classification":
            icon_path = os.path.join(assets_dir, "classification-dark.png")
        elif text == "Settings":
            icon_path = os.path.join(assets_dir, "settings-dark.png")
        
        # Create composite icon with an empty space
        original_icon = QPixmap(icon_path)
    
        combined = QPixmap(19 + 9, 19)  # 18px icon + 11px space
        combined.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(combined)
        painter.drawPixmap(0, 0, original_icon.scaled(19, 19, Qt.AspectRatioMode.KeepAspectRatio, 
                                                    Qt.TransformationMode.SmoothTransformation))
        painter.end()
        
        button.setIcon(QIcon(combined))
        button.setIconSize(QSize(19 + 9, 19))  # Combined icon with empty space

        if page_index == 0:
            button.setChecked(True)
        
        button.clicked.connect(lambda: self._switch_page(page_index, button))
        
        return button

    def _switch_page(self, index, clicked_button):
        """Switch between model controls and image classification pages"""
        self.content_stack.setCurrentIndex(index)
        
        # Update button state - find all buttons and uncheck them except the clicked one
        for i in range(self.sidebar.layout().count()):
            item = self.sidebar.layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QPushButton):
                item.widget().setChecked(item.widget() == clicked_button)
        
    def _connect_tab_status_signals(self):
        """Connecting signals from each tab to the main status bar"""
        tabs = [self.simple_cnn_tab, self.resnet_tab, self.efficientnet_tab, self.vgg16_tab]
        for tab in tabs:
            tab.status_message.connect(self.update_status_bar)

    def update_status_bar(self, message, timeout=8000):
        self.status_bar.showMessage(message, timeout)
             
    def _classify_all(self):
        """Classification of the loaded image using all loaded models"""
        if not self.image_classification_widget.current_image_path:
            self.update_status_bar("No image loaded for classification")
            return
            
        model_map = {
            self.simple_cnn_tab: {'type': 'simple_cnn', 'name': 'Simple CNN'},
            self.resnet_tab: {'type': 'resnet', 'name': 'ResNet'},
            self.efficientnet_tab: {'type': 'efficientnet', 'name': 'EfficientNet'},
            self.vgg16_tab: {'type': 'vgg16', 'name': 'VGG16'}
        }
        
        results = []
        for tab, model_info in model_map.items():
            if tab.model_loaded:
                try:
                    # Get image prediction
                    result = tab.model.predict_image(self.image_classification_widget.current_image_path)
                    predicted_class = result['class']
                    probabilities = result['probabilities']
                    
                    # Update result
                    self.image_classification_widget.update_result(model_info['type'], predicted_class)
                    
                    # Update plot
                    self.image_classification_widget.update_plot(model_info['type'], tab.model.classes, probabilities)
                    
                    results.append(f"{model_info['name']}: {predicted_class}")

                    self.update_status_bar("Classification complete")
                except Exception as e:
                    self.image_classification_widget.update_result(model_info['type'], "Error")
                    print(f"Error in {model_info['name']} classification: {str(e)}")
                    self.update_status_bar("No models available for classification")
            else:
                self.image_classification_widget.update_result(model_info['type'], "No model")
                
    def eventFilter(self, obj, event):
        """Event filter to handle window resize events"""
        
        if obj is self and event.type() == QEvent.Type.Resize:
            # Delay the adjustment slightly to avoid constant resizing during drag
            QTimer.singleShot(10, self._adjust_sidebar_width)
            
        return super().eventFilter(obj, event)
    
    def _adjust_sidebar_width(self):
        """Adjust sidebar width based on window width"""
        window_width = self.width()
        
        # Hide sidebar completely if window is too small
        if window_width < self.SIDEBAR_HIDE_THRESHOLD:
            self.sidebar_scroll.setVisible(False)
            
            # Adjusting the layout to use the full width of the content
            if self.main_layout.indexOf(self.sidebar_scroll) != -1:
                # Temporarily disconnect the sidebar from the layout
                self.main_layout.removeWidget(self.sidebar_scroll)
        else:
            # Check if the sidebar is visible, if not, add it back
            if not self.sidebar_scroll.isVisible():
                self.sidebar_scroll.setVisible(True)
                
                # Add the sidebar back to the layout if it is not there
                if self.main_layout.indexOf(self.sidebar_scroll) == -1:
                    content_widget = self.main_layout.itemAt(0).widget()
                    self.main_layout.removeWidget(content_widget)
                    self.main_layout.addWidget(self.sidebar_scroll, 1)
                    self.main_layout.addWidget(content_widget, 5)
            
            # Calculate the proportional width within the min/max limits
            # Window width from 800 to 1600 corresponds to sidebar width from 300 to 400
            proportion = min(1.0, (window_width - self.SIDEBAR_HIDE_THRESHOLD) / 800)
            sidebar_width = int(self.MIN_SIDEBAR_WIDTH + proportion * 
                               (self.MAX_SIDEBAR_WIDTH - self.MIN_SIDEBAR_WIDTH))
            
            # Ensures that the width is within the limits
            sidebar_width = max(self.MIN_SIDEBAR_WIDTH, 
                               min(self.MAX_SIDEBAR_WIDTH, sidebar_width))
            
            # Apply the width
            self.sidebar_scroll.setMinimumWidth(sidebar_width)
            self.sidebar_scroll.setMaximumWidth(sidebar_width)