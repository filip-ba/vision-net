from PyQt6.QtWidgets import QWidget, QHBoxLayout


class CenteredContentContainer(QWidget):
    """
    A ontainer widget that centers its content and limits maximum width.
    The content will not expand beyond the max_width and will stay centered
    when the window is resized.
    """
    def __init__(self, max_width=1200, parent=None):
        super().__init__(parent)
        
        # Main layout with outer margins set to 0
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create content container with specified max width
        self.content = QWidget()
        self.content.setMaximumWidth(max_width)
        self.content.setMinimumWidth(400)  
        
        # Add the content container centered in the layout
        self.main_layout.addStretch(1)
        self.main_layout.addWidget(self.content)
        self.main_layout.addStretch(1)
    
    def set_layout(self, layout):
        self.content.setLayout(layout)
        
    def get_content_widget(self):
        return self.content