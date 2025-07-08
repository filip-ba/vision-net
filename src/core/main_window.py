from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, QSizePolicy, 
                             QSpacerItem, QStatusBar, QFrame, QHBoxLayout, QStackedWidget, 
                             QPushButton, QLabel, QScrollArea)
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtCore import Qt, QSize, QTimer, QEvent
import os

from src.core.model_tab import ModelTab
from src.core.classification_tab import ClassificationTab
from src.core.settings_tab import SettingsTab
from src.core.dataset_tab import DatasetTab  
from ..models.simple_cnn_model import SimpleCnnModel  
from ..models.resnet_model import ResNetModel 
from ..models.efficientnet_model import EfficientNetModel  
from ..models.vgg16_model import VGG16Model
from ..utils.get_project_root import get_project_root


class MainWindow(QMainWindow):
    
    def __init__(self, style_manager):
        super().__init__()
        
        self.style_manager = style_manager
        
        # Sidebar width settings
        self.MIN_SIDEBAR_WIDTH = 190
        self.MAX_SIDEBAR_WIDTH = 260
        self.SIDEBAR_HIDE_THRESHOLD = 1000
        
        # Event filter for window resize
        self.installEventFilter(self)
        
        self._create_ui()
        self._set_icons_based_on_current_theme()

        self._connect_model_tab_status_signals()


    def _connect_model_tab_status_signals(self):
        """Connect status signals from tabs to the main window status bar"""
        self.simple_cnn_tab.status_message.connect(self.update_status_bar)
        self.resnet_tab.status_message.connect(self.update_status_bar)
        self.efficientnet_tab.status_message.connect(self.update_status_bar)
        self.vgg16_tab.status_message.connect(self.update_status_bar)

    def update_status_bar(self, message, timeout=8000):
        self.status_bar.showMessage(message, timeout)

    def _classify_all(self):
        """Classify the loaded image"""
        if not self.classification_tab.current_image_path:
            self.update_status_bar("No image loaded for classification")
            return
            
        model_map = {
            self.simple_cnn_tab: {'type': 'simple_cnn', 'name': 'Simple CNN'},
            self.resnet_tab: {'type': 'resnet', 'name': 'ResNet'},
            self.efficientnet_tab: {'type': 'efficientnet', 'name': 'EfficientNet'},
            self.vgg16_tab: {'type': 'vgg16', 'name': 'VGG16'}
        }
        
        for tab, model_info in model_map.items():
            if tab.model_loaded:
                try:
                    # Get image prediction
                    result = tab.model.predict_image(self.classification_tab.current_image_path)
                    predicted_class = result['class']
                    probabilities = result['probabilities']
                    self.classification_tab.update_result(model_info['type'], predicted_class)
                    self.classification_tab.update_plot(model_info['type'], tab.model.classes, probabilities)
                    self.update_status_bar("Classification complete")
                except Exception as e:
                    self.classification_tab.update_result(model_info['type'], "Error")
                    print(f"Error in {model_info['name']} classification: {str(e)}")
                    self.update_status_bar("No models available for classification")
            else:
                self.classification_tab.update_result(model_info['type'], "No model")
                self.classification_tab.init_plot(model_info['type'])

    def _switch_page(self, index, clicked_button):
        """Switch between sidebar pages"""
        self.content_stack.setCurrentIndex(index)
        
        # Update button state - find all buttons and uncheck them except the clicked one
        for i in range(self.sidebar.layout().count()):
            item = self.sidebar.layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QPushButton):
                item.widget().setChecked(item.widget() == clicked_button)
    
    def _set_icons_based_on_current_theme(self):
        project_root = get_project_root()
        icons_dir = os.path.join(project_root, "assets", "icons")
        theme_suffix = "light" if self.style_manager.get_current_style() == self.style_manager.STYLE_DARK else "dark"

        self._set_sidebar_icons(icons_dir, theme_suffix)
        self._set_menu_toggle_icon(icons_dir, theme_suffix)
        self.classification_tab.classification_widget.set_arrow_icons(icons_dir, theme_suffix)

    def _set_sidebar_icons(self, icons_dir, theme_suffix):
        for i in range(self.sidebar.layout().count()):
            item = self.sidebar.layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QPushButton):
                button = item.widget()
                button_text = button.text()
                
                icon_path = None
                if button_text == "Models":
                    icon_path = os.path.join(icons_dir, f"model-{theme_suffix}.png")
                elif button_text == "Classification":
                    icon_path = os.path.join(icons_dir, f"classification-{theme_suffix}.png")
                elif button_text == "Dataset":
                    icon_path = os.path.join(icons_dir, f"dataset-{theme_suffix}.png")
                elif button_text == "Settings":
                    icon_path = os.path.join(icons_dir, f"settings-{theme_suffix}.png")
                    
                if icon_path and os.path.exists(icon_path):
                    # Create composite icon with an empty space (looks better)
                    original_icon = QPixmap(icon_path)
                
                    combined = QPixmap(19 + 9, 19)  # 19px icon + 9px space
                    combined.fill(Qt.GlobalColor.transparent)
                    
                    painter = QPainter(combined)
                    painter.drawPixmap(0, 0, original_icon.scaled(19, 19, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    painter.end()
                    
                    button.setIcon(QIcon(combined))
                    button.setIconSize(QSize(19 + 9, 19))

    def _set_menu_toggle_icon(self, icons_dir, theme_suffix):
        menu_icon_path = os.path.join(icons_dir, f"menu-{theme_suffix}.png")
        if os.path.exists(menu_icon_path):
            menu_icon = QPixmap(menu_icon_path)
            self.sidebar_toggle_button.setIcon(QIcon(menu_icon))
            self.sidebar_toggle_button.setIconSize(QSize(16, 16))

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
            
            # Show the toggle button
            self.sidebar_toggle_button.setVisible(True)
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
            sidebar_width = max(self.MIN_SIDEBAR_WIDTH, min(self.MAX_SIDEBAR_WIDTH, sidebar_width))
            
            self.sidebar_scroll.setMinimumWidth(sidebar_width)
            self.sidebar_scroll.setMaximumWidth(sidebar_width)
            
            # Hide the toggle button when sidebar is visible
            self.sidebar_toggle_button.setVisible(False)

    def _toggle_sidebar(self):
        """Toggle sidebar visibility when the toggle button is clicked"""
        if self.sidebar_scroll.isVisible():
            # Hide sidebar
            self.sidebar_scroll.setVisible(False)
            if self.main_layout.indexOf(self.sidebar_scroll) != -1:
                self.main_layout.removeWidget(self.sidebar_scroll)
        else:
            # Show sidebar
            self.sidebar_scroll.setVisible(True)
            if self.main_layout.indexOf(self.sidebar_scroll) == -1:
                content_widget = self.main_layout.itemAt(0).widget()
                self.main_layout.removeWidget(content_widget)
                self.main_layout.addWidget(self.sidebar_scroll, 1)
                self.main_layout.addWidget(content_widget, 5)

    def eventFilter(self, obj, event):
        """Handle window resize events with event filter"""   
        if obj is self and event.type() == QEvent.Type.Resize:
            # Delay the adjustment slightly to avoid constant resizing during drag
            QTimer.singleShot(10, self._adjust_sidebar_width)
            
        return super().eventFilter(obj, event)
    
    def _create_ui(self):
        self.setWindowTitle("VisionNet")
        self.setGeometry(50, 50, 1000, 600)
        project_root = get_project_root()
        icons_dir = os.path.join(project_root, "assets", "icons")
        window_icon_path = os.path.join(icons_dir, "app-icon.png")        
        self.setWindowIcon(QIcon(window_icon_path))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar = self._create_sidebar()
        self.sidebar_scroll.setWidget(self.sidebar)
        self.sidebar_scroll.setMinimumWidth(self.MIN_SIDEBAR_WIDTH)
        self.sidebar_scroll.setMaximumWidth(self.MIN_SIDEBAR_WIDTH)
        self.sidebar_toggle_button = QPushButton()
        self.sidebar_toggle_button.setObjectName("sidebar-toggle-button")
        self.sidebar_toggle_button.setFixedSize(30, 30)
        self.sidebar_toggle_button.setVisible(False)
        self.sidebar_toggle_button.clicked.connect(self._toggle_sidebar)
        
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        toggle_button_layout = QHBoxLayout()
        toggle_button_layout.setContentsMargins(10, 10, 0, 0)
        toggle_button_layout.addWidget(self.sidebar_toggle_button)
        toggle_button_layout.addStretch()
        content_layout.addLayout(toggle_button_layout)

        # Create stacked widget for switching between settings and classification
        self.content_stack = QStackedWidget()
        
        # Create model settings (first page)
        self.tab_widget = QTabWidget()
        self.tab_widget.setMovable(True)
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  

        self.simple_cnn_tab = ModelTab(model_class=SimpleCnnModel)
        self.resnet_tab = ModelTab(model_class=ResNetModel)
        self.efficientnet_tab = ModelTab(model_class=EfficientNetModel)
        self.vgg16_tab = ModelTab(model_class=VGG16Model)
        
        self.tab_widget.addTab(self.simple_cnn_tab, "Simple CNN")
        self.tab_widget.addTab(self.resnet_tab, "ResNet18")
        self.tab_widget.addTab(self.efficientnet_tab, "EfficientNet-B0")
        self.tab_widget.addTab(self.vgg16_tab, "VGG16")
        
        # Create image classification (second page)
        self.classification_tab = ClassificationTab()
        self.classification_tab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create dataset tab (third page)
        model_tabs = [self.simple_cnn_tab, self.resnet_tab, self.efficientnet_tab, self.vgg16_tab]
        self.dataset_tab = DatasetTab(model_tabs)
        self.dataset_tab.status_message.connect(self.update_status_bar)
        # Refresh classification test images when new dataset is loaded
        self.dataset_tab.dataset_loaded.connect(self.classification_tab.load_test_images)  
        self.dataset_tab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create settings page (fourth page)
        self.settings_tab = SettingsTab(self.style_manager)
        self.settings_tab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.settings_tab.style_changed.connect(self._set_icons_based_on_current_theme)

        # Create scroll areas for each page
        models_scroll = QScrollArea()
        models_scroll.setWidgetResizable(True)
        models_scroll.setFrameShape(QFrame.Shape.NoFrame)
        models_scroll.setWidget(self.tab_widget)

        classification_scroll = QScrollArea()
        classification_scroll.setWidgetResizable(True)
        classification_scroll.setFrameShape(QFrame.Shape.NoFrame)
        classification_scroll.setWidget(self.classification_tab)

        dataset_scroll = QScrollArea()
        dataset_scroll.setWidgetResizable(True)
        dataset_scroll.setFrameShape(QFrame.Shape.NoFrame)
        dataset_scroll.setWidget(self.dataset_tab)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setWidget(self.settings_tab)

        # Add scroll areas to the content stack
        self.content_stack.addWidget(models_scroll)
        self.content_stack.addWidget(classification_scroll)
        self.content_stack.addWidget(dataset_scroll)
        self.content_stack.addWidget(settings_scroll)

        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("status-bar")
        
        content_layout.addWidget(self.content_stack, 1)
        content_layout.addWidget(self.status_bar, 0)
        
        self.main_layout.addWidget(self.sidebar_scroll, 1)
        self.main_layout.addWidget(content_container, 5)  
        
        self.classification_tab.classify_clicked.connect(self._classify_all) 
        self.classification_tab.image_loaded.connect(self.update_status_bar) 

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
        
        title_label = QLabel("VisionNet")
        title_label.setObjectName("app-title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(title_label)
        
        sidebar_layout.addWidget(logo_widget)

        # Add custom separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setObjectName("sidebar-separator")

        sidebar_layout.addWidget(separator)

        # Add empty space under the separator 
        spacer = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sidebar_layout.addItem(spacer)
        
        # Add navigation buttons
        models_btn = self._create_sidebar_button("Models", 0)
        classify_btn = self._create_sidebar_button("Classification", 1)
        dataset_btn = self._create_sidebar_button("Dataset", 2)
        
        sidebar_layout.addWidget(models_btn)
        sidebar_layout.addWidget(classify_btn)
        sidebar_layout.addWidget(dataset_btn)
        
        # Add stretch to push the settings button to the bottom
        sidebar_layout.addStretch()

        # Add separator above settings button
        bottom_separator = QFrame()
        bottom_separator.setFrameShape(QFrame.Shape.HLine)
        bottom_separator.setFrameShadow(QFrame.Shadow.Plain)
        bottom_separator.setObjectName("sidebar-separator")
        sidebar_layout.addWidget(bottom_separator)
        
        sidebar_layout.addItem(spacer)

        # Add Settings button at the bottom
        settings_btn = self._create_sidebar_button("Settings", 3)
        sidebar_layout.addWidget(settings_btn)

        sidebar_layout.addItem(spacer)
        
        return sidebar
    
    def _create_sidebar_button(self, text, page_index):
        button = QPushButton(text)
        button.setObjectName("sidebar-button")
        button.setCheckable(True)

        if page_index == 0:
            button.setChecked(True)
        
        button.clicked.connect(lambda: self._switch_page(page_index, button))
        
        return button