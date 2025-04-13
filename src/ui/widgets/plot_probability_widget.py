from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QPushButton, QStackedWidget, QSizePolicy
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from ...utils.custom_canvas import ScrollableFigureCanvas


class PlotProbabilityWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_plot_button = None
        self.model_names = {
            'simple_cnn': 'Simple CNN',
            'resnet': 'ResNet',
            'efficientnet': 'EfficientNet',
            'vgg16': 'VGG16'
        }
        self.plot_widgets = {}
        self.plot_titles = {}
        self.plot_buttons = {}
        self._create_ui()
        
    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("plot-3-frame")
        self.plot_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        frame_layout = QVBoxLayout(self.plot_frame)
        
        # Plot title labels
        self.plot_titles = {}
        for model_type, title in self.model_names.items():
            title_label = QLabel(f"{title} Class Probabilities")
            title_label.setObjectName("plot-label")
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
        
        frame_layout.addWidget(self.plot_stack)
        frame_layout.setSpacing(9)

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
        layout.addWidget(self.plot_frame)
        
        self.setLayout(layout)
        
        # Initialize empty plots
        self.init_plot()
        self.switch_plot('simple_cnn')
        
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
            ax.set_title('')
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
                f'{height:.2%}',
                ha="center", va="bottom")

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('Probability', fontsize=9, labelpad=15)
        ax.tick_params(axis='both', labelsize=9)
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        figure.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.9)
        figure.tight_layout()

        self.plot_widgets[model_type]['canvas'].draw() 