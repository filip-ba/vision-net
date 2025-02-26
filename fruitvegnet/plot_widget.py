from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from fruitvegnet.utils.scrollable_figure_canvas import ScrollableFigureCanvas


class PlotWidget(QWidget):

    def __init__(self, title, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Frame styled like a group box
        self.frame = QFrame()
        self.frame.setObjectName("StyledFrame")  
        self.frame.setStyleSheet("""
            QFrame#StyledFrame {
                border: 1px solid #b0b0b0;
                border-radius: 8px;
                padding: 5px;
                background-color: white;
            }
        """)
        frame_layout = QVBoxLayout(self.frame)
        main_layout.addWidget(self.frame)

        # Title label
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.title_label)

        # Figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = ScrollableFigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_layout.addWidget(self.canvas)
        
        # Explicitly apply the style to the frame
        self.update_style()
        
    def update_style(self):
        self.frame.setStyleSheet("""
            QFrame#StyledFrame {
                border: 1px solid #b0b0b0;
                border-radius: 8px;
                padding: 5px;
                background-color: white;
            }
        """)
        
    def showEvent(self, event):
        """Update styles when widget is displayed"""
        super().showEvent(event)
        self.update_style()
        
    def resizeEvent(self, event):
        """Update styles when changing widget size"""
        super().resizeEvent(event)
        self.update_style()

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
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
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
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
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