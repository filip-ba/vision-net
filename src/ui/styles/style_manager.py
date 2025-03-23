import os
from PyQt6.QtCore import QSettings


class StyleManager:
    """Manages application styles and provides functionality to switch between themes."""
    
    STYLE_LIGHT = "light"
    STYLE_DARK = "dark"
    
    def __init__(self, app, default_style=STYLE_LIGHT):
        self.app = app
        self.settings = QSettings("FruitVegNet", "AppSettings")
        
        # Load saved style or use the default one
        self.current_style = self.settings.value("style", default_style)
        self.apply_style(self.current_style)
    
    def load_stylesheet(self, filename):
        """Load a stylesheet from a file."""
        current_file_path = os.path.abspath(__file__)
        styles_dir = os.path.dirname(current_file_path)
        path = os.path.join(styles_dir, filename)
        with open(path, "r") as f:
            return f.read()
    
    def apply_style(self, style_name):
        """Apply the specified style to the application."""
        self.current_style = style_name
        global_style = self.load_stylesheet("global.css")
        theme_style = self.load_stylesheet(f"{style_name}.css")
        combined_style = global_style + theme_style
        self.app.setStyleSheet(combined_style)
        self.settings.setValue("style", style_name)
        
    def toggle_style(self):
        """Toggle between light and dark styles."""
        if self.current_style == self.STYLE_LIGHT:
            self.apply_style(self.STYLE_DARK)
        else:
            self.apply_style(self.STYLE_LIGHT)
    
    def get_current_style(self):
        """Get the name of the current style."""
        return self.current_style