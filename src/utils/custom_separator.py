from PyQt6.QtWidgets import QFrame

def create_separator(orientation):
    separator = QFrame()

    if orientation == "horizontal":
        frame_shape = QFrame.Shape.HLine    
    else:
        frame_shape = QFrame.Shape.VLine

    separator.setObjectName("main-separator")
    separator.setFrameShape(frame_shape)
    separator.setFrameShadow(QFrame.Shadow.Plain)
    separator.setLineWidth(0)
    separator.setMidLineWidth(0)

    return separator