from PyQt6.QtWidgets import QFrame

def create_separator():
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setFrameShadow(QFrame.Shadow.Plain)
    separator.setLineWidth(0)
    separator.setMidLineWidth(0)
    separator.setStyleSheet(
        "color: #e3e3e3; "
        "height: 1px; "
        "margin: 0px; "
        "padding: 0px;"
    )
    return separator