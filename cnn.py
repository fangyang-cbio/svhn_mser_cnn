import os
import torch
import numpy as np 
import torchvision
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split
import torch.nn as nn
from torch.nn import AvgPool2d
import torch.nn.functional as F 
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader

import dataprep
from dataprep import DataPrep

#Code for Google Colab
'''from google.colab import drive
drive.mount('/content/drive')
model_save_name = 'svhn_cnn_format1.pth'
path = F"/content/drive/My Drive/cs6476/{model_save_name}" '''

#code for local run
DATA_SET = 'dataset_svhn'
INPUT_DIR = "input_images"
OUTPUT_DIR = "graded_images"
MODEL = 'svhn_cnn_format1.pth'

'''Code skeleton ref to https://github.com/nanekja/JovianML-Project/blob/master/SVHN_Dataset_Classification_v2.ipynb but model is different'''

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class classifier(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'test_loss': loss.detach(), 'test_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['test_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['test_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(
            epoch, result['train_loss'], result['test_loss'], result['test_acc']))


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, test_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, test_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

class CnnModel(classifier):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            nn.Conv2d(256, 11, kernel_size=1, stride=1, padding=1),
            
            nn.AvgPool2d(kernel_size=6),           
            nn.Flatten(),
            #nn.Softmax(dim=1)
            )
        
    def forward(self, xb):
        return self.network(xb)

if torch.cuda.is_available():
    print('GPU')
    device = torch.device('cuda')
else:
    print('CPU')
    device = torch.device('cpu')

if not os.path.exists(MODEL):
    try:
        x = np.load(os.path.join(DATA_SET,'train_img.npy'))
        y = np.load(os.path.join(DATA_SET,'train_lable.npy'))
        xt = np.load(os.path.join(DATA_SET,'test_img.npy'))
        yt = np.load(os.path.join(DATA_SET,'test_lable.npy'))
    except:
        data = DataPrep()
        data.download()
        data.getnpy('train')
        data.getnpy('test')
        x, y, xt, yt = data.x, data.y, data.xt, data.yt


    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.001

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    xt = torch.Tensor(xt)
    yt = torch.Tensor(yt)

    train = TensorDataset(x,y.long())
    train_dl = DataLoader(train, batch_size=128, shuffle=True)
    test = TensorDataset(xt,yt.long())
    test_dl = DataLoader(test, batch_size=128, shuffle=True)

    images_train, labels_train = next(iter(train_dl))
    images_test, labels_test = next(iter(test_dl))

    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    model = CnnModel()
    model = to_device(CnnModel(), device)
    
    fit(num_epochs, lr, model, train_dl, test_dl, opt_func)
    torch.save(model.state_dict(), svhn_cnn_format1.pth)