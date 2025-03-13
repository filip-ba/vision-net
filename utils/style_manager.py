import os


class StyleManager:
    """Manages application styles and provides functionality to switch between themes."""
    
    STYLE_LIGHT = "light"
    STYLE_DARK = "dark"
    
    def __init__(self, app, default_style=STYLE_LIGHT):
        self.app = app
        self.current_style = default_style
        self.project_root = self._get_project_root()
        self.apply_style(self.current_style)
    
    def _get_project_root(self):
        """Get the project root directory."""
        # Assuming this file is in the utils folder
        current_file_path = os.path.abspath(__file__)
        utils_dir = os.path.dirname(current_file_path)
        return os.path.dirname(utils_dir)
    
    def load_stylesheet(self, filename):
        """Load a stylesheet from a file."""
        path = os.path.join(self.project_root, "utils", filename)
        with open(path, "r") as f:
            return f.read()
    
    def apply_style(self, style_name):
        """Apply the specified style to the application."""
        self.current_style = style_name
        
        # Load the global stylesheet
        global_style = self.load_stylesheet("style_global.css")
        
        # Load the theme-specific stylesheet
        theme_style = self.load_stylesheet(f"style_{style_name}.css")
        
        # Combine stylesheets
        combined_style = global_style + theme_style
        
        # Apply to application
        self.app.setStyleSheet(combined_style)
        
    def toggle_style(self):
        """Toggle between light and dark styles."""
        if self.current_style == self.STYLE_LIGHT:
            self.apply_style(self.STYLE_DARK)
        else:
            self.apply_style(self.STYLE_LIGHT)
    
    def get_current_style(self):
        """Get the name of the current style."""
        return self.current_style