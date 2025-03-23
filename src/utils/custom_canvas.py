from PyQt6.QtCore import QEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class ScrollableFigureCanvas(FigureCanvas):
    
    """Modified version of FigureCanvas to allow mouse scroll events to pass to parent widgets (QScrollArea)."""
    def __init__(self, figure):
        super().__init__(figure)
        self.setMouseTracking(True)
        
    def event(self, event):
        # If it is a mouse wheel event, forward it to the parent widget
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            return False
        return super().event(event)