from PyQt6.QtWidgets import ( 
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSlider, QDoubleSpinBox, 
    QSpinBox, QSizePolicy, QFileDialog,QScrollArea, QFrame, QStatusBar )
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImageReader, QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os


class PlotWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.setContentsMargins(0, 0, 0, 0)
        # Frame styled like a group box
        frame = QFrame()
        frame.setObjectName("StyledFrame")  
        frame.setStyleSheet("""
            QFrame#StyledFrame {
                border: 1px solid #c4c8cc;
                border-radius: 6px;
                padding: 5px;
                background-color: white;
            }
        """)
        frame_layout = QVBoxLayout(frame)
        main_layout.addWidget(frame)
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.title_label)
        # Figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_layout.addWidget(self.canvas)

    def plot(self, x, y, title=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        if title:
            self.title_label.setText(title)
        ax.grid(True)
        self.canvas.draw()


class ParameterWidget(QWidget):
    def __init__(self, label, min_val, max_val, default_val, decimals=0, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        param_label = QLabel(label)
        param_label.setMinimumWidth(100)
        layout.addWidget(param_label)
        # SpinBox setup
        if decimals == 0:
            self.spinbox = QSpinBox()
        else:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(default_val)
        self.spinbox.setFixedWidth(80)
        layout.addWidget(self.spinbox)
        # Slider setup
        self.slider = QSlider(Qt.Orientation.Horizontal)
        if decimals > 0:
            self.slider.setRange(0, int(max_val * (10 ** decimals)))
            self.slider.setValue(int(default_val * (10 ** decimals)))
        else:
            self.slider.setRange(int(min_val), int(max_val))
            self.slider.setValue(int(default_val))
        layout.addWidget(self.slider)
        # Connect signals
        self._setup_connections()

    def _setup_connections(self):
        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)
    
    def _slider_changed(self, value):
        if isinstance(self.spinbox, QDoubleSpinBox):
            self.spinbox.setValue(value / (10 ** self.spinbox.decimals()))
        else:
            self.spinbox.setValue(value)
    
    def _spinbox_changed(self, value):
        if isinstance(self.spinbox, QDoubleSpinBox):
            self.slider.setValue(int(value * (10 ** self.spinbox.decimals())))
        else:
            self.slider.setValue(int(value))


class ImageClassificationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        # Image display area
        self.image_display = QLabel()
        self.image_display.setFixedSize(300, 300)
        self.image_display.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Load placeholder image
        self._load_placeholder()
        # Controls area
        controls_layout = QVBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.setStyleSheet("padding: 8px;")
        self.classify_btn = QPushButton("Classify Image")
        self.classify_btn.setStyleSheet("padding: 8px;")
        self.result_label = QLabel("Classification: None")
        self.result_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        controls_layout.addWidget(self.load_image_btn)
        controls_layout.addWidget(self.classify_btn)
        controls_layout.addWidget(self.result_label)
        controls_layout.addStretch()
        # Add to main layout
        layout.addWidget(self.image_display)
        layout.addLayout(controls_layout)

    def _load_placeholder(self):
        placeholder_path = os.path.join(os.path.dirname(__file__), "..", "placeholder_img.jpg")
        if os.path.exists(placeholder_path):
            pixmap = QPixmap(placeholder_path)
            self.image_display.setPixmap(pixmap.scaled(
                300, 300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.image_display.setText("No image loaded")

    def update_image(self, pixmap):
        """Method for updating the displayed image"""
        if pixmap:
            scaled_pixmap = pixmap.scaled(
                300, 300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
        else:
            self.image_display.setText("No image loaded")


class PytorchTab(QWidget):
    def __init__(self):
        super().__init__()
        self._create_ui()
        self.current_image_path = None
        self._setup_connections()  

    def _setup_connections(self):
        self.image_widget.load_image_btn.clicked.connect(self.load_image)
        self.image_widget.classify_btn.clicked.connect(self.classify_image)
        self.load_model_btn.clicked.connect(self.load_model)
        self.train_model_btn.clicked.connect(self.train_model)
        self.test_model_btn.clicked.connect(self.test_model)

    def load_image(self):
        """Method for loading an image"""
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
                self.image_widget.update_image(pixmap)
                self.image_widget.result_label.setText("Classification: None")
                self.status_bar.showMessage(f"Loaded image: {os.path.basename(file_path)}", 3000)
            else:
                self.image_widget.image_display.setText("Failed to load image")
                self.current_image_path = None
                self.status_bar.showMessage("Failed to load image", 3000)

    def classify_image(self):
        """Method for image classification"""
        if self.current_image_path:
            # Zde přidejte vlastní logiku klasifikace
            self.image_widget.result_label.setText("Classification: Example Class")
            self.status_bar.showMessage("Classification complete", 3000)
        else:
            self.status_bar.showMessage("No image loaded", 3000)

    def load_model(self):
        """Method for loading the model"""
        self.status_bar.showMessage("Loading model...", 3000)
        # Zde přidejte vlastní logiku načtení modelu

    def train_model(self):
        """Method for model training"""
        self.status_bar.showMessage("Training model...", 3000)
        # Zde přidejte vlastní logiku trénování modelu

    def test_model(self):
        """Method for model testing"""
        self.status_bar.showMessage("Testing model...", 3000)
        # Zde přidejte vlastní logiku testování modelu

    def _create_ui(self):
        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)  
        # Left panel
        left_panel = self._create_left_panel()
        scroll_layout.addWidget(left_panel)
        # Right panel (plots)
        right_panel = self._create_right_panel()
        scroll_layout.addWidget(right_panel)
        # Set up scroll area
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        # Add status bar
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        main_layout.addWidget(self.status_bar)

    def _create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 15, 10)
        left_panel.setFixedWidth(500)
        # Model Controls
        model_group = QGroupBox("Model Controls")
        model_layout = QVBoxLayout()
        # Model buttons
        buttons_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model")
        self.train_model_btn = QPushButton("Train Model")
        self.test_model_btn = QPushButton("Test Model")
        for btn in [self.load_model_btn, self.train_model_btn, self.test_model_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    padding: 8px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:disabled {
                    background-color: #e9ecef;
                    color: #6c757d;
                }
            """)
            buttons_layout.addWidget(btn)
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #dc3545;
                padding: 5px;
            }
        """)
        model_layout.addLayout(buttons_layout)
        model_layout.addWidget(self.model_status)
        model_group.setLayout(model_layout)
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        # Create parameter widgets
        self.epochs_widget = ParameterWidget("Epochs:", 1, 1000, 10)
        self.learning_rate_widget = ParameterWidget("Learning Rate:", 0.000001, 1.0, 0.001, 6)
        self.momentum_widget = ParameterWidget("Momentum:", 0.0, 1.0, 0.9, 6)
        for widget in [self.epochs_widget, self.learning_rate_widget, self.momentum_widget]:
            params_layout.addWidget(widget)
        params_group.setLayout(params_layout)
        # Image Classification
        self.image_widget = ImageClassificationWidget()
        image_group = QGroupBox("Image Classification")
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_widget)
        image_group.setLayout(image_layout)
        # Add all components to left panel
        for widget in [model_group, params_group, image_group]:
            widget.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #c4c8cc;
                    border-radius: 6px;
                    margin-top: 20px;
                    padding: 5px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 0px;
                    padding: 0 3px 0 3px;
                }
            """)
            left_layout.addWidget(widget)
        left_layout.addStretch()
        return left_panel

    def _create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(15, 30, 10, 10) 
        # Create plot widgets with titles
        self.plot_widget1 = PlotWidget("Training Loss")
        self.plot_widget2 = PlotWidget("Validation Accuracy")
        for plot in [self.plot_widget1, self.plot_widget2]:
            plot.setMinimumHeight(300)
            right_layout.addWidget(plot)
        # Example plots
        x = range(10)
        y1 = [i**2 for i in x]
        y2 = [i**3 for i in x]
        self.plot_widget1.plot(x, y1)
        self.plot_widget2.plot(x, y2)
        right_layout.addStretch()
        return right_panel