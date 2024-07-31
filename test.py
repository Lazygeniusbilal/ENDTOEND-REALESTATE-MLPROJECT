# test_imports.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from mlProject.utils.common import read_yaml, create_directories

print("Imports successful!")
