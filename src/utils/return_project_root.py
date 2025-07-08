import os
import sys

def get_project_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        current_file_path = os.path.abspath(__file__)
        core_dir = os.path.dirname(current_file_path)
        src_dir = os.path.dirname(core_dir)
        project_root = os.path.dirname(src_dir)
        return project_root