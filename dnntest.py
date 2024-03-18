
import os
from model import ResNet, BasicBlock, available_volumes, available_percents

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim
import pandas as pd
from PIL import Image

batch_size = 48

num_classes_to_learn = len(available_volumes) * len(available_percents) + 3


celoss = nn.CrossEntropyLoss(reduction='sum')
l2loss = nn.MSELoss(reduction='mean')


def create_classes_array(_list_size, _certain_class):
    result = [0.0] * _list_size
    if _certain_class is not None:
        result[_certain_class] = 1.0
    return np.array(result)

class ImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.data = []

        for root, subdirs, files in os.walk(root_dir):
            if not subdirs:
                try:
                    x, volume, percent = root.split('/')
                    for file in files:
                        self.data.append({'type': 'bottle', 'volume': volume, 'percent': percent, 'filename': f"{root}/{file}"})
                except:
                    x, xtype = root.split('/')
                    for file in files:
                        self.data.append({'type': xtype, 'filename': f"{root}/{file}"})

        # print(self.data)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xtype = self.data[idx]['type']

        if xtype == 'bottle':
            volume = float(self.data[idx]['volume'])
            percent = float(self.data[idx]['percent'])
            class_to_learn = available_volumes.index(volume) * len(available_percents) + available_percents.index(percent)
            img_hash = create_classes_array(num_classes_to_learn, class_to_learn)
        elif xtype == 'incident':
            img_hash = create_classes_array(num_classes_to_learn, num_classes_to_learn - 1)
        elif xtype == 'hands':
            img_hash = create_classes_array(num_classes_to_learn, num_classes_to_learn - 2)
        elif xtype == 'empty':
            img_hash = create_classes_array(num_classes_to_learn, num_classes_to_learn - 3)
        else:
            img_hash = create_classes_array(num_classes_to_learn, None)

        img_name = self.data[idx]['filename']
        # print(img_name)
        img = Image.open(img_name).convert('RGB')
        t = self.transform(img).clone().detach()
        sample = [t, img_hash, img_name]
        return sample


def train(net, data_size=100, steps=100):
    trainset = ImagesDataset(root_dir='dataset', transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

    for epoch in range(steps):  # loop over the dataset multiple times
        # print(',', end='')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, names = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = l2loss(outputs, labels) * celoss(outputs, labels)

            if i == 0:
                print(inputs)
                print(names[0])
                print(['%.4f' % elem for elem in outputs[0].tolist()])
                print(['%.4f' % elem for elem in labels[0].tolist()])

                print(torch.argmax(outputs[0]).item())
                print(torch.argmax(labels[0]).item())

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print(loss.item())

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.8f}')

            running_loss = 0.0
        # torch.save(net, 'mynet')
    print('Finished Training')


device = "cuda"

preprocess = Compose([
    ToTensor()
])

print(f"Using {device} device")

# mynet = torch.load('_mynet')
torch
mynet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes_to_learn)
mynet.to(device)

print(mynet)
train(mynet, data_size=100, steps=1000)
