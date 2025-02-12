from PyQt6.QtWidgets import ( 
    QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy )
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

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

    def plot_confusion_matrix(self, plot_widget, conf_mat = None, classes = None):
        plot_widget.figure.clear()
        plot_widget.figure.subplots_adjust(left=0.25, right=0.85, bottom=0.25, top=0.85)
        plot_widget.figure.tight_layout()
        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel('Predicted', labelpad=10)
        ax.set_ylabel('True', labelpad=10)
        # Empty confusion matrix
        if conf_mat is None or classes is None:
            ax.set_title('(Will be populated after testing)', pad=10)
        else:   # Populated confusion matrix
            im = ax.imshow(conf_mat, cmap='Blues', aspect='auto')
            cbar = plot_widget.figure.colorbar(im)
            cbar.ax.tick_params(labelsize=8)
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, conf_mat[i, j], 
                        ha="center", va="center",
                        color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")         
        plot_widget.canvas.draw()

    def plot_loss_history(self, plot_widget, epochs = None, train_loss_history = None, val_loss_history = None):
        plot_widget.figure.clear()
        plot_widget.figure.subplots_adjust(left=0.25, right=0.85, bottom=0.25, top=0.85)
        plot_widget.figure.tight_layout()
        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel('Epochs', labelpad=10)
        ax.set_ylabel('Loss', labelpad=10)
        ax.grid(True)
        if epochs is not None or train_loss_history is not None or val_loss_history is not None:
            ax.plot(range(1, epochs + 1), train_loss_history, label='Training Loss')
            ax.plot(range(1, epochs + 1), val_loss_history, label='Validation Loss')
            ax.legend()       
        plot_widget.canvas.draw()