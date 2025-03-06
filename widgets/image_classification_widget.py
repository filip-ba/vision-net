from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QFrame,
    QFileDialog, QLabel, QPushButton, QStackedWidget, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImageReader, QIcon, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from matplotlib.figure import Figure
import os, random
from matplotlib import pyplot as plt

from utils.scrollable_figure_canvas import ScrollableFigureCanvas


class ImageClassificationWidget(QWidget):
    image_loaded = pyqtSignal(str, int)  # Signal to notify when new image is loaded
    classify_clicked = pyqtSignal()  # Signal for classify button clicks

    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.active_plot_button = None  # Track currently active plot button
        
        # Define model names first (moved up from below)
        self.model_names = {
            'simple_cnn': 'Simple CNN',
            'resnet': 'ResNet',
            'efficientnet': 'EfficientNet',
            'vgg16': 'VGG16'
        }
        
        self._create_ui()
        
    def _create_ui(self):
        # Main group box
        main_group = QGroupBox("Image Classification")
        main_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout = QVBoxLayout(main_group)
        
        # Top section - Image, buttons, and result labels
        top_layout = QHBoxLayout()
        
        # Left side - Image and buttons
        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Image preview
        self.image_display = QLabel()
        self.image_display.setFixedSize(150, 150)
        self.image_display.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.image_display)
        
        # Buttons
        button_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.classify_btn = QPushButton("Classify All")
        for btn in [self.load_btn, self.classify_btn]:
            button_layout.addWidget(btn)
        left_layout.addLayout(button_layout)
        left_layout.addStretch()
        
        # Middle - Result labels with Show Plot buttons
        middle_layout = QVBoxLayout()
        middle_widget = QWidget()
        middle_widget.setLayout(middle_layout)
        middle_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        self.result_labels = {}
        self.plot_buttons = {}
        
        # Define label style based on MetricsWidget
        self.label_style = """
            QLabel {
                font-size: 14px;
                padding: 8px;
                background-color: #f0f0f0;
                border: 1px solid #bbb;
                border-radius: 5px;
                font-weight: 500;
            }
        """

        # Define the button styles
        self.button_normal_style = """
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """
        
        self.button_pressed_style = """
            QPushButton {
                background-color: #88c2ff;
                border: 1px solid #495057;
                border-radius: 4px;
            }
        """
        
        for model_type, label_text in self.model_names.items():
            # Create container for each model
            container = QWidget()
            container_layout = QHBoxLayout(container)
            
            # Create and style label with new style
            label = QLabel(f"{label_text}: None")
            label.setStyleSheet(self.label_style)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setFixedSize(200, 40)
            container_layout.addWidget(label)
            
            # Icon path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            icon_path = os.path.join(project_root, "assets", "graph_icon.png")
            pixmap = QPixmap(icon_path)

            # Create plot button
            plot_btn = QPushButton()
            icon = QIcon(pixmap) 
            plot_btn.setIcon(icon)
            plot_btn.setIconSize(QSize(32, 32)) 
            plot_btn.setFixedSize(40, 40)
            plot_btn.setStyleSheet(self.button_normal_style)
            plot_btn.setCheckable(True)  
            
            container_layout.addWidget(plot_btn)
            
            # Store references
            self.result_labels[model_type] = label
            self.plot_buttons[model_type] = plot_btn
            
            middle_layout.addWidget(container)
        
        middle_layout.addStretch()
        
        # Add widgets to top layout
        top_layout.addWidget(left_widget, 1)
        top_layout.addWidget(middle_widget, 2)
        
        # Bottom section - Plot Widget with fixed size
        bottom_layout = QVBoxLayout()
        
        # Create StyledFrame
        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("StyledFrame")
        self.plot_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_frame.setMaximumHeight(400)  # Set maximum height for the plot
        frame_layout = QVBoxLayout(self.plot_frame)
        
        # Create the title labels for each plot 
        self.plot_titles = {}
        for model_type, title in self.model_names.items():
            title_label = QLabel(f"{title} - Class Probabilities")
            title_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setVisible(False)  # Initially hidden
            self.plot_titles[model_type] = title_label
            frame_layout.addWidget(title_label)

        # Create plot stack
        self.plot_stack = QStackedWidget()
        self.plot_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create plot widgets for each model
        self.plot_widgets = {}
        
        for model_type in self.model_names.keys():
            # Create figure with fixed size ratio
            figure = Figure(figsize=(5, 4), dpi=100)
            canvas = ScrollableFigureCanvas(figure)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.plot_widgets[model_type] = {'figure': figure, 'canvas': canvas}
            self.plot_stack.addWidget(canvas)
            
            # Initialize empty plot
            self.init_plot(model_type)
            
            # Connect button to show this plot
            self.plot_buttons[model_type].clicked.connect(
                lambda checked, m=model_type: self.switch_plot(m)
            )
        
        frame_layout.addWidget(self.plot_stack)
        bottom_layout.addWidget(self.plot_frame)
        
        # Add both layouts to main layout
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)
        
        # Main widget layout
        widget_layout = QVBoxLayout(self)
        widget_layout.addWidget(main_group)
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_image)
        self.classify_btn.clicked.connect(self.classify_clicked.emit)
        
        # Load initial image from dataset
        self.load_random_test_image()
        
        # Set the first button as active by default
        self.switch_plot('simple_cnn')
        
    def switch_plot(self, model_type):
        """Switch to the specified plot and update button states"""
        # Set the current widget in the stack
        self.plot_stack.setCurrentWidget(self.plot_widgets[model_type]['canvas'])
        
        # Update title visibility
        for plot_type, title in self.plot_titles.items():
            title.setVisible(plot_type == model_type)
        
        # Update button states
        for btn_type, btn in self.plot_buttons.items():
            if btn_type == model_type:
                # Check the current button and apply pressed style
                btn.setChecked(True)
                btn.setStyleSheet(self.button_pressed_style)
                self.active_plot_button = btn
            else:
                # Uncheck other buttons and restore normal style
                btn.setChecked(False)
                btn.setStyleSheet(self.button_normal_style)
        
    def init_plot(self, model_type=None):
        """Initialize empty probability plot for specified or all models"""
        if model_type:
            models = [model_type]
        else:
            models = self.plot_widgets.keys()
            
        for model in models:
            figure = self.plot_widgets[model]['figure']
            figure.clear()
            ax = figure.add_subplot(111)
            
            ax.set_title("")
            ax.set_xlabel('')
            ax.set_ylabel('Probability', fontsize=9, labelpad=15)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_ylim(0, 1)
            
            # Consistent layout settings
            figure.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.9)
            figure.tight_layout()
            self.plot_widgets[model]['canvas'].draw()

    def update_plot(self, model_type, classes, probabilities):
        """Update probability plot for a specific model"""
        if model_type not in self.plot_widgets:
            return
            
        figure = self.plot_widgets[model_type]['figure']
        figure.clear()
        ax = figure.add_subplot(111)
        bars = ax.bar(classes, probabilities)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0%}',
                ha="center", va="bottom")

        ax.set_title("")
        ax.set_xlabel('')
        ax.set_ylabel('Probability', fontsize=9, labelpad=15)
        ax.tick_params(axis='both', labelsize=9)
        
        # Dynamic y-axis limit based on data and model type
        max_prob = max(probabilities)
        if model_type == 'simple_cnn':
            # For Simple CNN, set dynamic y-limit with 20% padding above the highest bar
            y_max = min(1.0, max_prob * 1.2)
            ax.set_ylim(0, max(y_max, 0.5))  # Ensure minimum height of 0.5 for visibility
        else:
            # For more accurate models, maintain consistent 0-1.2 scale
            ax.set_ylim(0, 1.2)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Consistent layout settings
        figure.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.9)
        figure.tight_layout()
        self.plot_widgets[model_type]['canvas'].draw()

    def load_random_test_image(self):
        """Load a random image from the test dataset"""
        dataset_path = "./dataset/fruit_dataset/test"
        if os.path.exists(dataset_path):
            # Get all class directories
            class_dirs = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
            
            if class_dirs:
                # Choose random class
                random_class = random.choice(class_dirs)
                class_path = os.path.join(dataset_path, random_class)
                
                # Get all images in the class directory
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    # Choose random image
                    random_image = random.choice(images)
                    image_path = os.path.join(class_path, random_image)
                    
                    # Load the image
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        self.current_image_path = image_path
                        scaled_pixmap = pixmap.scaled(
                            150, 150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self.image_display.setPixmap(scaled_pixmap)
                        return
                    
        # If we get here, something went wrong
        self.image_display.setText("No image")
        self.current_image_path = None
            
    def load_image(self):
        supported_formats = [f"*.{fmt.data().decode()}" for fmt in QImageReader.supportedImageFormats()]
        filter_string = "Image Files ({})".format(" ".join(supported_formats))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            filter_string
        )
        
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.current_image_path = file_path
                scaled_pixmap = pixmap.scaled(
                    150, 150,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_display.setPixmap(scaled_pixmap)
                self.image_loaded.emit("Image loaded", 8000)
            else:
                self.image_display.setText("Failed to load image")
                self.image_loaded.emit("Failed to load image", 8000)
                self.current_image_path = None
                
    def update_result(self, model_type, result):
        """Update the classification result for a specific model"""
        if model_type in self.result_labels:
            model_name = self.model_names[model_type]
            self.result_labels[model_type].setText(f"{model_name}: {result}")