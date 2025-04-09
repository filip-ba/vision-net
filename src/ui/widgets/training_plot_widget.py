from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np


class TrainingPlotWidget(QWidget):

    def __init__(self, title, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("plot-label")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        spacer = QWidget()
        spacer.setFixedHeight(9)
        spacer.setObjectName("spacer")

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.figure.patch.set_facecolor('white')
        self.canvas.setStyleSheet("background-color: white;")
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: white;")

        main_layout.addWidget(self.title_label)
        main_layout.addWidget(spacer)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

    def plot_confusion_matrix(self, plot_widget, conf_mat = None, classes = None):
        plot_widget.figure.clear()
        plot_widget.figure.subplots_adjust(left=0.25, right=0.85, bottom=0.35, top=0.85)
        plot_widget.figure.tight_layout()
        
        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel('Predicted', labelpad=15)
        ax.set_ylabel('True', labelpad=15)

        if conf_mat is not None or classes is not None:
            im = ax.imshow(conf_mat, cmap='Blues', aspect='auto')
            cbar = plot_widget.figure.colorbar(im)
            cbar.ax.tick_params(labelsize=8)
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes, fontsize=9)
            ax.set_yticklabels(classes, fontsize=9)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, conf_mat[i, j], 
                        ha="center", va="center",
                        color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")  
                        
        plot_widget.canvas.draw()

    def plot_loss_history(self, plot_widget, epochs=None, train_loss_history=None, val_loss_history=None):
        plot_widget.figure.clear()
        ax = plot_widget.figure.add_subplot(111)
        plot_widget.figure.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.9)
        ax.clear()
        ax.set_xlabel('Epochs', labelpad=15)
        ax.set_ylabel('Loss', labelpad=15)
        ax.grid(True)

        if train_loss_history is not None and val_loss_history is not None:
            x = list(range(1, len(train_loss_history) + 1)) if epochs is None else list(range(1, epochs + 1))
            
            # In case of single epoch, plot points instead of lines
            if len(train_loss_history) == 1:
                ax.plot(x, train_loss_history, 'bo-', label='Training Loss', markersize=8)  
                ax.plot(x, val_loss_history, 'ro-', label='Validation Loss', markersize=8)  
                ax.set_xticks([1]) 
            else:
                ax.plot(x, train_loss_history, 'b-', label='Training Loss')
                ax.plot(x, val_loss_history, 'r-', label='Validation Loss')
            ax.legend()
            
        plot_widget.canvas.draw() 