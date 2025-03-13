from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QFrame,
    QFileDialog, QLabel, QPushButton, QStackedWidget, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImageReader, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from matplotlib.figure import Figure
import os, random
from matplotlib import pyplot as plt

from utils.scrollable_figure_canvas import ScrollableFigureCanvas
from utils.custom_separator import create_separator


class ImageClassification(QWidget):
    image_loaded = pyqtSignal(str, int)  # Signal for image load events
    classify_clicked = pyqtSignal()  # Signal for classify button clicks

    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.active_plot_button = None
        
        # Model names mapping
        self.model_names = {
            'simple_cnn': 'Simple CNN',
            'resnet': 'ResNet',
            'efficientnet': 'EfficientNet',
            'vgg16': 'VGG16'
        }
        
        self._create_ui()    

    def _create_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 0)
        main_layout.setSpacing(0)

        # Create top and bottom layouts
        top_layout = self._create_top_layout()
        bottom_layout = self._create_bottom_layout()
        
        # Add layouts to main layout with proportions
        main_layout.addLayout(top_layout, 4)
        main_layout.addLayout(bottom_layout, 6)
        
        self.setLayout(main_layout)
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_image)
        self.classify_btn.clicked.connect(self.classify_clicked.emit)
        
        # Initial setup
        self.load_random_test_image()
        self.switch_plot('simple_cnn')
        
    def _create_top_layout(self):
        # Top section for image, buttons and results
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # Image classification section
        image_layout_group = self._create_image_section()
        
        # Results section
        results_group = self._create_results_section()
        
        # Add widgets to top layout
        top_layout.addWidget(image_layout_group, 2)
        top_layout.addWidget(results_group, 3)
        
        return top_layout
        
    def _create_image_section(self):
        # Create image and buttons group
        image_layout_group = QGroupBox("Image Classification")
        image_layout_group.setObjectName("classification-group")
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(15, 15, 15, 15)
        image_layout.setSpacing(15)
        
        # Image display
        self.image_display = QLabel()
        self.image_display.setObjectName("image-display")
        self.image_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_display.setMinimumSize(100, 100)  
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.image_display)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        
        self.load_btn = QPushButton("Load")
        self.classify_btn = QPushButton("Classify")
        
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.classify_btn)
        
        image_layout.addLayout(button_layout)
        image_layout_group.setLayout(image_layout)
        
        return image_layout_group
        
    def _create_results_section(self):
        # Results section with model names and predictions
        results_group = QGroupBox("Results")
        results_group.setObjectName("results-group")
        results_main_layout = QVBoxLayout(results_group)
        results_main_layout.setContentsMargins(0, 18, 0, 18)
        
        results_columns_layout = QHBoxLayout()
        results_columns_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left column - model names
        left_column = QVBoxLayout()
        left_column.setSpacing(18)
        left_column.setContentsMargins(0, 0, 0, 0)

        # Right column - results
        right_column = QVBoxLayout()
        right_column.setSpacing(18)
        right_column.setContentsMargins(0, 0, 0, 0)

        # Vertical separator between columns
        vertical_separator = QFrame()
        vertical_separator.setFrameShape(QFrame.Shape.VLine)
        vertical_separator.setFrameShadow(QFrame.Shadow.Plain)
        vertical_separator.setStyleSheet(
            "color: #e3e3e3; "
            "width: 1px; "
            "margin: 0px; "
            "padding: 0px;"
        )

        # Create model name labels and result labels
        model_labels = {
            'simple_cnn': QLabel("Simple CNN"),
            'resnet': QLabel("ResNet"),
            'efficientnet': QLabel("EfficientNet"),
            'vgg16': QLabel("VGG 16")
        }
        
        # Configure model name labels
        for model_id, label in model_labels.items():
            label.setObjectName(f"Model{model_id.title().replace('_', '')}")
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            label.setMinimumWidth(70)
            left_column.addWidget(label, 1)
            
            # Add separator after each label except the last one
            if model_id != list(self.model_names.keys())[-1]:
                left_column.addWidget(create_separator("horizontal"))

        # Create result labels
        self.result_labels = {}
        for i, model_id in enumerate(self.model_names.keys()):
            result_label = QLabel("None") 
            result_label.setStyleSheet("font-weight: 700;")
            result_label.setObjectName("ModelResultLabel")
            result_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            result_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            result_label.setMinimumWidth(70)
            self.result_labels[model_id] = result_label
            right_column.addWidget(result_label, 1)
            
            # Add separator after each except the last one
            if model_id != list(self.model_names.keys())[-1]:
                right_column.addWidget(create_separator("horizontal"))
        
        # Assemble results layout
        results_columns_layout.addLayout(left_column)
        results_columns_layout.addWidget(create_separator("vertical"))
        results_columns_layout.addLayout(right_column)
        results_main_layout.addLayout(results_columns_layout)
        
        results_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        results_group.setMinimumWidth(100)

        return results_group
    
    def _create_bottom_layout(self):
        # Bottom section with plots
        bottom_layout = QVBoxLayout()
        
        # Create plot frame
        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("plot-3-frame")
        self.plot_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_layout = QVBoxLayout(self.plot_frame)
        
        # Plot title labels
        self.plot_titles = {}
        for model_type, title in self.model_names.items():
            title_label = QLabel(f"{title} - Class Probabilities")
            title_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setVisible(False)
            self.plot_titles[model_type] = title_label
            frame_layout.addWidget(title_label)

        # Plot stack widget
        self.plot_stack = QStackedWidget()
        self.plot_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_stack.setMinimumWidth(200)

        # Create plot widgets for each model
        self.plot_widgets = {}
        for model_type in self.model_names.keys():
            figure = Figure(figsize=(5, 4), dpi=100)
            canvas = ScrollableFigureCanvas(figure)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumWidth(180)
            self.plot_widgets[model_type] = {'figure': figure, 'canvas': canvas}
            self.plot_stack.addWidget(canvas)
            self.init_plot(model_type)
        
        frame_layout.addWidget(self.plot_stack)
        frame_layout.setSpacing(0)

        # Plot selection buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(0)
        
        self.plot_buttons = {}
        for model_type, label_text in self.model_names.items():
            btn = QPushButton(label_text)
            btn.setCheckable(True)
            btn.setObjectName(f"plot-{model_type}")
            self.plot_buttons[model_type] = btn
            buttons_layout.addWidget(btn, 1)
            
            # Connect button to show the corresponding plot
            btn.clicked.connect(
                lambda checked, m=model_type: self.switch_plot(m)
            )
        
        frame_layout.addLayout(buttons_layout)
        bottom_layout.addWidget(self.plot_frame)
        
        return bottom_layout
        
    def switch_plot(self, model_type):
        """Switch to the specified plot and update button states"""
        self.plot_stack.setCurrentWidget(self.plot_widgets[model_type]['canvas'])
        
        # Update title visibility
        for plot_type, title in self.plot_titles.items():
            title.setVisible(plot_type == model_type)
        
        # Update button states
        for btn_type, btn in self.plot_buttons.items():
            if btn_type == model_type:
                btn.setChecked(True)
                self.active_plot_button = btn
            else:
                btn.setChecked(False)
        
    def init_plot(self, model_type=None):
        """Initialize empty probability plot"""
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

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0%}',
                ha="center", va="bottom")

        ax.set_title("")
        ax.set_xlabel('')
        ax.set_ylabel('Probability', fontsize=9, labelpad=15)
        ax.tick_params(axis='both', labelsize=9)
        
        # Dynamic y-axis limit based on data
        max_prob = max(probabilities)

        if model_type == 'simple_cnn':
            # Dynamic y-limit with padding for Simple CNN
            y_max = min(1.0, max_prob * 1.2)
            ax.set_ylim(0, max(y_max, 0.5))
        else:
            # Consistent scale for more accurate models
            ax.set_ylim(0, 1.2)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        figure.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.9)
        figure.tight_layout()
        self.plot_widgets[model_type]['canvas'].draw()

    def load_random_test_image(self):
        """Load a random image from the test dataset"""
        dataset_path = "./dataset/fruit_dataset/test"
        if os.path.exists(dataset_path):
            class_dirs = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
            
            if class_dirs:
                random_class = random.choice(class_dirs)
                class_path = os.path.join(dataset_path, random_class)
                
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    random_image = random.choice(images)
                    image_path = os.path.join(class_path, random_image)
                    
                    self.current_image_path = image_path
                    self.update_image_display(image_path)
                    return
                    
        self.image_display.setText("No image")
        self.current_image_path = None
            
    def load_image(self):
        """Load image from file system"""
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
                self.update_image_display(file_path)
                self.image_loaded.emit("Image loaded", 8000)
            else:
                self.image_display.setText("Failed to load image")
                self.image_loaded.emit("Failed to load image", 8000)
                self.current_image_path = None
                
    def update_image_display(self, image_path):
        """Update the image display with the specified image"""
        if not image_path:
            return
            
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
            
        self.original_pixmap = pixmap
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
        """Handle resize events"""
        super().resizeEvent(event)
        self.scale_image()
                
    def update_result(self, model_type, result):
        """Update the classification result for a specific model"""
        if model_type in self.result_labels:
            self.result_labels[model_type].setText(result)

    def showEvent(self, event):
        """Handle the widget being shown for the first time"""
        super().showEvent(event)
        # Scale the image after the widget is visible and has proper dimensions
        QTimer.singleShot(50, self.scale_image)