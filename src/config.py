import matplotlib
# Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io

from sklearn.model_selection import train_test_split

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils

from .utils import JSONConfig

# Load config
config_file = os.getenv('MRCNN_CONFIG', "default_config.json")
CONFIGS = JSONConfig(**json.load(open(config_file, 'r')))

# Dataset directory
DATA_DIR = CONFIGS.dataset_dir

# Number of classes
CLASSES = CONFIGS.classes

# Get list of images
ALL_IMAGES, TRAIN_IMAGE_IDS, VAL_IMAGE_IDS = [], [], []
ALL_IMAGES = [folder for folder in os.listdir(DATA_DIR) if not folder.startswith(".")]
TRAIN_IMAGE_IDS, VAL_IMAGE_IDS = train_test_split(ALL_IMAGES, test_size=0.1, random_state=2019)

############################################################
#  Configurations
############################################################
