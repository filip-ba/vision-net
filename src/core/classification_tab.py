from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFileDialog)
from PyQt6.QtGui import QPixmap, QImageReader
from PyQt6.QtCore import pyqtSignal, QTimer
import os, random

from .classification_tab_widgets.classification_widget import ClassificationWidget
from .classification_tab_widgets.results_widget import ResultsWidget
from .classification_tab_widgets.plot_probability_widget import PlotProbabilityWidget
from ..utils.get_project_root import get_project_root


class ClassificationTab(QWidget):
    image_loaded = pyqtSignal(str, int)  
    classify_clicked = pyqtSignal()  

    def __init__(self):
        super().__init__()  

        self._create_ui()    
        
        self.current_image_path = None
        self.classification_results_cache = {}
        self.test_images = []
        self.current_test_index = -1
        self._load_placeholder_image()

    def load_test_images(self, dataset_path=None):
        self.test_images = []

        dataset_dir = os.path.join(dataset_path, "test") if dataset_path else None

        if dataset_dir and os.path.exists(dataset_dir):
            for class_name in os.listdir(dataset_dir):
                class_path = os.path.join(dataset_dir, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_name)
                            self.test_images.append((img_path, class_name))
        else:
            return False

        random.shuffle(self.test_images)
        self.current_test_index = 0
        self._show_current_test_image(suppress_message=True)

    def _load_placeholder_image(self):
        project_root = get_project_root()
        placeholder_path = os.path.join(project_root, "assets", "themes", "classification-placeholder.jpg")
        if os.path.exists(placeholder_path):
            self.classification_widget.update_image(placeholder_path)
            return False

    def show_next_test_image(self):       
        if self.current_test_index < len(self.test_images) - 1:
            self.current_test_index += 1
            self._show_current_test_image()
            
    def show_previous_test_image(self):
        if self.current_test_index >= 0:
            self.current_test_index -= 1
            self._update_navigation_buttons()
            self._show_current_test_image()

    def _show_current_test_image(self, suppress_message=False):
        if 0 <= self.current_test_index < len(self.test_images):
            image_path, class_name = self.test_images[self.current_test_index]
            self.current_image_path = image_path
            self.classification_widget.update_image(image_path)
            self._load_results_from_cache(image_path, class_name)
            self._update_navigation_buttons()
            if not suppress_message:
                self.image_loaded.emit(f"Showing test image {self.current_test_index + 1}/{len(self.test_images)} - {class_name}", 3000)

    def _update_navigation_buttons(self):
        prev_enabled = self.current_test_index > 0
        next_enabled = self.current_test_index < len(self.test_images) - 1
        self.classification_widget.update_navigation_buttons(prev_enabled, next_enabled)
        
    def _load_results_from_cache(self, image_path, class_name=None):
        if image_path in self.classification_results_cache:
            cached_results = self.classification_results_cache[image_path]
            self.results_widget.reset_results()
            self.plot_widget.init_plot()
            
            for model_type, data in cached_results.items():
                if 'result' in data:
                    self.update_result(model_type, data['result'], from_cache=True)
                if 'plot' in data:
                    classes, probs = data['plot']
                    self.update_plot(model_type, classes, probs, from_cache=True)
            
            if cached_results:
                first_model = next(iter(cached_results))
                self.plot_widget.switch_plot(first_model)
        else:
            self.plot_widget.init_plot()
            self.results_widget.reset_results()
            if not class_name:
                self.image_loaded.emit("Image loaded, previous results cleared.", 8000)
            
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
                self.classification_widget.update_image(file_path)
                self._load_results_from_cache(file_path)
            else:
                self.current_image_path = None
                self.image_loaded.emit("Failed to load image", 8000)
                
    def update_result(self, model_type, result, from_cache=False):
        if not from_cache and self.current_image_path:
            if self.current_image_path not in self.classification_results_cache:
                self.classification_results_cache[self.current_image_path] = {}
            if model_type not in self.classification_results_cache[self.current_image_path]:
                self.classification_results_cache[self.current_image_path][model_type] = {}
            self.classification_results_cache[self.current_image_path][model_type]['result'] = result
            
        self.results_widget.update_result(model_type, result)

    def update_plot(self, model_type, classes, probabilities, from_cache=False):
        if not from_cache and self.current_image_path:
            if self.current_image_path not in self.classification_results_cache:
                self.classification_results_cache[self.current_image_path] = {}
            if model_type not in self.classification_results_cache[self.current_image_path]:
                self.classification_results_cache[self.current_image_path][model_type] = {}
            self.classification_results_cache[self.current_image_path][model_type]['plot'] = (classes, probabilities)
            
        self.plot_widget.update_plot(model_type, classes, probabilities)
        
    def init_plot(self, model_type=None):
        self.plot_widget.init_plot(model_type)
        
    def switch_plot(self, model_type):
        self.plot_widget.switch_plot(model_type)

    def showEvent(self, event):
        """Scale the image in image display properly after the start of the application"""
        super().showEvent(event)
        QTimer.singleShot(50, self.classification_widget.scale_image)

    def _create_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 0)
        main_layout.setSpacing(10)  
        top_layout = self._create_top_layout()
        bottom_layout = self._create_bottom_layout()
        main_layout.addLayout(top_layout, 4)
        main_layout.addLayout(bottom_layout, 6)
        self.setLayout(main_layout)

        self.classification_widget.load_img_btn_clicked.connect(self.load_image)
        self.classification_widget.classify_img_btn_clicked.connect(self.classify_clicked.emit)
        self.classification_widget.prev_clicked.connect(self.show_previous_test_image)
        self.classification_widget.next_clicked.connect(self.show_next_test_image)
        
    def _create_top_layout(self):
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self.classification_widget = ClassificationWidget()
        self.results_widget = ResultsWidget()
        
        top_layout.addWidget(self.classification_widget, 2)
        top_layout.addWidget(self.results_widget, 2)
        return top_layout
        
    def _create_bottom_layout(self):
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self.plot_widget = PlotProbabilityWidget()
        self.plot_widget.setMinimumHeight(300)
        
        bottom_layout.addWidget(self.plot_widget)
        return bottom_layout