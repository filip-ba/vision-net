from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QToolBar, QToolButton
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import numpy as np
import os, sys


class CustomToolbar(QToolBar):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.nav = NavigationToolbar2QT(canvas, parent)
        self.setIconSize(QSize(25, 25))
        self.nav.setObjectName("matplotlib-toolbar")

        self.tool_buttons = {}

        project_root = self.get_project_root()
        icon_dir = os.path.join(project_root, "assets", "icons", "mpl_toolbar_icons")

        buttons = [
            ("home", "home.png", self.nav.home, False),
            ("back", "back.png", self.nav.back, False),
            ("forward", "forward.png", self.nav.forward, False),
            ("move", "move.png", self.nav.pan, True),
            ("zoom", "zoom.png", self.nav.zoom, True),
            ("qt4_editor_options", "qt4_editor_options.png", self.nav.edit_parameters, False),
            ("subplots", "subplots.png", self.nav.configure_subplots, False),
            ("filesave", "filesave.png", self.nav.save_figure, False),
        ]

        for name, icon_file, callback, checkable in buttons:
            icon_path = os.path.join(icon_dir, icon_file)
            self.add_button(name, icon_path, callback, checkable)

    def add_button(self, name, icon_path, callback, checkable=False):
        btn = QToolButton(self)
        btn.setObjectName("matplotlib-toolbutton")
        btn.setIcon(QIcon(icon_path))
        btn.setToolTip(name.capitalize().replace('_', ' '))
        btn.setCheckable(checkable)
        btn.clicked.connect(lambda checked=False, b=btn: self.on_button_clicked(name, callback, b))
        self.tool_buttons[name] = btn
        self.addWidget(btn)

    def on_button_clicked(self, name, callback, clicked_button):
        """Disable all checkable buttons except the current one"""
        for n, b in self.tool_buttons.items():
            if b.isCheckable() and b != clicked_button:
                b.setChecked(False)
        callback()

    def get_project_root(self):
        """Returns the path to the root directory of the project, works both in development and in the executable"""
        if getattr(sys, 'frozen', False):
            # Executable
            return sys._MEIPASS
        else:
            # IDE
            current_file_path = os.path.abspath(__file__)
            widgets_dir = os.path.dirname(current_file_path)
            ui_dir = os.path.dirname(widgets_dir)
            src_dir = os.path.dirname(ui_dir)
            project_root = os.path.dirname(src_dir)
            return project_root


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

        self.toolbar = CustomToolbar(self.canvas, self)
        self.toolbar.setObjectName("matplotlib-toolbar")

        main_layout.addWidget(self.title_label)
        main_layout.addWidget(spacer)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

    def plot_confusion_matrix(self, plot_widget, conf_mat=None, classes=None):
        plot_widget.figure.clear()
        plot_widget.figure.subplots_adjust(left=0.25, right=0.85, bottom=0.35, top=0.85)
        plot_widget.figure.tight_layout()

        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel('Predicted', labelpad=15)
        ax.set_ylabel('True', labelpad=15)

        # Limit to a maximum of 10 classes
        if conf_mat is not None and classes is not None:
            conf_mat = conf_mat[:10, :10]  # Limit confusion matrix to 10x10
            classes = classes[:10]  # Limit classes to 10

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
            min_len = min(len(train_loss_history), len(val_loss_history))
            if min_len > 0:
                if len(train_loss_history) == 1:
                    x_train = [0]
                    x_val = [0]
                else:
                    epochs_actual = len(train_loss_history) - 1
                    x_train = [i * epochs_actual / (len(train_loss_history) - 1) for i in range(len(train_loss_history))]
                    x_val = [i * epochs_actual / (len(val_loss_history) - 1) for i in range(len(val_loss_history))]

                ax.plot(x_train, train_loss_history, 'b-', label='Training Loss')
                ax.plot(x_val, val_loss_history, 'r-', label='Validation Loss')
                ax.set_xticks(range(epochs_actual + 1))
                ax.legend()

        plot_widget.canvas.draw()