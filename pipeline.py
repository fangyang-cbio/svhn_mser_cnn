import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

import cnn
from cnn import CnnModel

INPUT_DIR = "input_images"
OUTPUT_DIR = "graded_images"
MODEL = 'svhn_cnn_format1.pth'

'''In grraded_images, image 1 is from https://www.budgetmailboxes.com/address-plaques-and-numbers, others are from Zillow.'''


class Pipeline:
    def __init__(self, image, image_file):
        self.image_file = image_file
        self.image = image
        self.regions = []
        self.candidates = []
        self.predicts = []

    def detect_regions(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        #mser.setMinArea(20)
        #mser.setMaxArea(800)
        mser.setDelta(5)
        msers, bboxes = mser.detectRegions(image_gray)
        grouped, w = cv2.groupRectangles(bboxes.tolist(), 1)
        for rectangle in grouped:
            if (rectangle[3]/rectangle[2] > 1.2 and rectangle[3]/rectangle[2] < 3) or (rectangle[2]/rectangle[3] > 1.2 and rectangle[2]/rectangle[3] < 3):
                self.regions.append(rectangle)

    def mser_draw(self,file):
        image = self.image.copy()
        for region in self.regions:
            x, y, w, h = region
            tl = (x, y)
            br = (x+w, y+h)
            cv2.rectangle(image, tl, br, (255, 0, 0))
        cv2.imshow('mser_draw', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite(os.path.join('mser_draw', file),image)

    
    def extraNresize(self, rw = 32, rh = 32):
        for region in self.regions:
            x, y, w, h = region;
            croped = self.image[y: (y+h), x: (x+w)]
            resized = cv2.resize(croped, (rw, rh))
            if w/h < 1:
                self.candidates.append(resized)
            else:
                resized = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.candidates.append(resized)

    def predict(self, model):
        l = len(self.candidates)
        candidates = np.array(self.candidates, dtype=np.float32)
        candidates = np.swapaxes(candidates,2,3)
        candidates = np.swapaxes(candidates,1,2)
        candidate = torch.from_numpy(candidates)
        outputs = model(candidate)
        outputs = F.softmax(outputs, dim =1)
        self.predicts = outputs.cpu().detach().numpy()

    def draw_predict(self):
        image = self.image.copy()
        digits = []
        for i in range(len(self.regions)):
            if np.max(self.predicts[i]) > 0.95:
                x, y, w, h = self.regions[i]
                tl = (x, y)
                br = (x+w, y+h)
                cv2.rectangle(image, tl, br, (255, 0, 0))
                n = np.argmax(self.predicts[i])%10
                cv2.putText(image, str(n), (x+w+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                digits.append([self.regions[i],n])
        digits = sorted(digits, key=lambda x: x[0][0]+x[0][1])
        result = 'Found: '
        for digit in digits:
            result = result + str(digit[1])
        result = result  + ' in ' + self.image_file
        print(result)
        cv2.imshow('mser_draw', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_predict(self, image_file):
        image = self.image.copy()
        for i in range(len(self.regions)):
            if np.max(self.predicts[i]) > 0.95:
                #print(self.predicts[i])
                x, y, w, h = self.regions[i]
                tl = (x, y)
                br = (x+w, y+h)
                cv2.rectangle(image, tl, br, (255, 0, 0))
                n = np.argmax(self.predicts[i])%10
                cv2.putText(image, str(n), (x+w+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, image_file),image)




