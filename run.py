import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

import cnn
from cnn import CnnModel

import pipeline
from pipeline import Pipeline

INPUT_DIR = "input_images"
OUTPUT_DIR = "graded_images"
MODEL = './svhn_cnn_format1.pth'


imagesFiles = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]

model = CnnModel()
model.load_state_dict(torch.load(MODEL, map_location=torch.device('cpu')))

for image_file in imagesFiles:
    image = cv2.imread(os.path.join(INPUT_DIR, image_file), cv2.IMREAD_COLOR)
    process = Pipeline(image, image_file)
    process.detect_regions()
    process.mser_draw(image_file)
    process.extraNresize()
    process.predict(model)
    process.draw_predict()
    process.save_predict(image_file)


