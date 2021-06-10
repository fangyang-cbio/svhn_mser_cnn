import os
import sys
import numpy
import scipy.io 

import gzip
import tarfile
import h5py
import cv2
from PIL import Image

import six.moves.cPickle as pickle 
from six.moves import urllib
import shutil

'''
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/cs6476"'''

DATA_SET = 'dataset_svhn'

train_adr = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
test_adr = 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'

class DataPrep:
    def __init__(self):
        self.x = []
        self.y = []
        self.xt = []
        self.yt = []

    def bboxHelper(self, attr):
        if (len(attr) > 1):
            attr = [f[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def download(self):
        train_file = os.path.join(DATA_DIR, 'train.tar.gz')
        test_file = os.path.join(DATA_DIR, 'test.tar.gz')

        urllib.request.urlretrieve(train_adr, train_file)
        urllib.request.urlretrieve(test_adr, test_file)

        train_tar = tarfile.open(train_file)
        test_tar = tarfile.open(test_file)

        train_tar.extractall(DATA_DIR)
        test_tar.extractall(DATA_DIR)

    def getnpy(self, name):
        MAT_FILE = os.path.join(DATA_DIR, name, 'digitStruct.mat')
        f = h5py.File(MAT_FILE, 'r')

        digitStructName = f['digitStruct']['name']
        digitStructBbox = f['digitStruct']['bbox']
        digitStructName

        image_dict = {}
        for i in range(len(digitStructName)):
            bbox = {}
            bb = digitStructBbox[n].item()
            bbox['label'] = self.bboxHelper(f[bb]["label"])
            bbox['left'] = self.bboxHelper(f[bb]["left"])
            bbox['top'] = self.bboxHelper(f[bb]["top"])
            bbox['height'] = self.bboxHelper(f[bb]["height"])
            bbox['width'] = self.bboxHelper(f[bb]["width"])
            bbox['label'] = bbox['label']+[10]
            w = min(bbox['left'])
            bbox['width'] = bbox['width']+ [w]
            h = min(bbox['top'])
            bbox['height'] = bbox['height']+ [h]
            bbox['left'] = bbox['left']+[0]
            bbox['top'] = bbox['top']+[0]
            image_dict[''.join([chr(c[0]) for c in f[digitStructName[i][0]].value])] = bbox

        names = []
        for item in os.listdir(os.path.join(DATA_DIR, name)):
            if item.endswith('.png'):
                names.append(item)

        y = []
        x = []
        #for i in range(3):
        for i in range(len(names)):
            path = os.path.join(DATA_DIR, name)
            image = cv2.imread(path + '/' + names[i])
            for j in range(len(image_dict[names[i]]['label'])):
                X = int(image_dict[names[i]]['left'][j])
                Y = int(image_dict[names[i]]['top'][j])
                w = int(image_dict[names[i]]['width'][j])
                h = int(image_dict[names[i]]['height'][j])
                img = image[Y:Y+h,X:X+w]
                try:
                    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
                    img_array = numpy.array(img, dtype=np.float32)
                    img_array1 = np.swapaxes(img_array,1,2)
                    img_array2 = np.swapaxes(img_array1,0,1)
                    y.append(image_dict[names[i]]['label'][j].astype(int))
                    x.append(img_array2)
                except:
                    continue
        
        if name == 'train':
            self.x = x
            self.y = y
        elif name == 'test':
            self.xt = x
            self.yt = y

        np.save(os.path.join(DATA_DIR, '{}_img.npy'.format(name)),np.asarray(x))
        np.save(os.path.join(DATA_DIR, '{}_lable.npy'.format(name)),np.asarray(y))







