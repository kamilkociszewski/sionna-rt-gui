import os

PROJECT_DIR = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
SOURCE_DIR = os.path.join(PROJECT_DIR, "src")
CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs", "sionna_rt_gui")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_CONFIG_PATH = os.path.join(CONFIGS_DIR, "base.yaml")

from . import gui
from .gui import SionnaRtGui
from .reload import AppHolder
