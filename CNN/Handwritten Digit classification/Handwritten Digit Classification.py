import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# setting a random seed to reproduce results
seed=4
tf.random.set_seed(4)

import cv2 as cv
import os

model = tf.keras.models.load_model("E:\DL-Tensorflow\Handwritten Digit classification\saved_model\my_model")

print(model.summary())

print(dir(model))