from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor    

# Create a light palette
palette = QPalette()
palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)