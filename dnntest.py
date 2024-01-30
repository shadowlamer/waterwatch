import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim
import pandas as pd
from PIL import Image

batch_size = 32

available_volumes = [1.5, 5, 12, 19]
available_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_classes_to_learn = len(available_volumes) * len(available_percents)

criterion = nn.CrossEntropyLoss(reduction='sum')


def create_classes_array(_list_size, _certain_class):
    result = [0.0] * _list_size
    result[_certain_class] = 1.0
    return np.array(result)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14
        x = self.layer4(x)  # 7x7

        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)  # remove 1 X 1 grid and make vector of tensor shape
        x = self.fc(x)

        return x.double()


class ImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.data = []

        for root, subdirs, files in os.walk(root_dir):
            if not subdirs:
                x, volume, percent = root.split('/')
                for file in files:
                    self.data.append({'volume': volume, 'percent': percent, 'filename': f"{root}/{file}"})

        # print(self.data)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data[idx]['filename']
        volume = float(self.data[idx]['volume'])
        percent = float(self.data[idx]['percent'])
        class_to_learn = available_volumes.index(volume) * len(available_percents) + available_percents.index(percent)
        img_hash = create_classes_array(num_classes_to_learn, class_to_learn)
        # print(img_name)
        img = Image.open(img_name).convert('RGB')
        t = self.transform(img).clone().detach()
        sample = [t, img_hash]

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
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print(loss.item())

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / data_size:.8f}')
            running_loss = 0.0
        torch.save(net, 'mynet')
    print('Finished Training')


device = "cuda"

preprocess = Compose([
    ToTensor()
])

print(f"Using {device} device")

# mynet = torch.load('mynet')

mynet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes_to_learn)

mynet.to(device)
print(mynet)
train(mynet, data_size=100, steps=1000)
