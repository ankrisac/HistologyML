import kivy
import tensorflow as tf
import cv2
import matplotlib as mp
import numpy as np

import os

current_dir = os.getcwd()
samples_dir = os.path.join(current_dir, 'samples')

if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)


app_creators = f"""
--- MicroLabs Software ---
Sachin Saravanan  - AI and lead dev
Maathangi Anirudh - Gridsplit dev and marketing
Samarth Chandna   - UI and marketing
Archie Parekh     - Biology Consultant
"""

app_credits = f"""
MicroLab - DeepStain

--- System ---

Kivy       - [{kivy.__version__}]
Tensorflow - [{tf.__version__}]
Numpy      - [{np.__version__}]
Matplotlib - [{mp.__version__}]
{app_creators}"""


interface = """
INTERFACE OVERVIEW

interface stuff WIP
"""

