from django.http import HttpResponse

from biomedicalcare.models import Patient
from biomedicalcare.models import Describe
import os
from django.core.files.storage import FileSystemStorage
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from django.contrib import messages


from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout as auth_logout
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels,
                  identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=1, num_classes=3):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def make_prediciton(image_path):
    image = Image.open(image_path).convert("L")
    pretrained_size = 224
    pretrained_means = [0.5]
    pretrained_stds = [0.5]

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    image = train_transforms(image)
    image = image.reshape(1, 1, 224, 224)
    model = ResNet50()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        r'ResNet50_Updated.pth', map_location=device))
    model.eval()

    predict = model(image)
    softmax = nn.Softmax(dim=1)
    predict = softmax(predict)
    probability, prediction = torch.max(predict, dim=1)

    # converting torch tensor into numpy array
    probability = probability.detach().numpy()[0]
    prediction = prediction.detach().numpy()[0]
    predict = predict.detach().numpy()[0]

    return probability, prediction, predict


@login_required(login_url="login")
def dashboard(request):
    if request.method == 'POST':
        # Get form data from request.POST dictionary
        firstname = request.POST['firstname']
        lastname = request.POST['lastname']
        sex = request.POST['sex']
        age = request.POST['age']
        address = request.POST['address']
        contactnumber = request.POST['contactnumber']
        date = request.POST['date']
        filename = request.FILES.get('filename')
        upload = request.FILES.get('filename')

        # Create new Patient instance and set its attributes
        patient = Patient(
            firstname=firstname,
            lastname=lastname,
            gender=sex,
            age=age,
            address=address,
            contactnumber=contactnumber,
            date=date,
            filename=filename
        )

        # Save the patient instance to the database
        patient.save()

        file_system_storage = FileSystemStorage()
        file_path = file_system_storage.save(upload.name, upload)

        # file_ = 'images/' + upload.name
        file_url = file_system_storage.url(file_path)
        print("file_path variable: ", file_path)

        probability, prediction, predict = make_prediciton(
            os.path.join('images', file_path))

        # predict = predict * 100
        predict = [round(i, 5) for i in predict]
        print("predicted probs", predict)

        f = open("prediction.txt", "w")
        f.write("probabiloty: " + str(probability))
        f.write("\n")
        f.write("prediction: " + str(prediction))
        f.close()

        classify = ""
        if prediction == 0:
            classify = "Benign"
        elif prediction == 1:
            classify = "Malignant"
        else:
            classify = "Normal"
        data = Describe.objects.all().first()
        if prediction == 0:
            desc = data.Benign
        elif prediction == 1:
            desc = data.Malignant
        else:
            desc = data.Normal

        userss = Patient.objects.get(contactnumber=contactnumber)
        print(userss)
        userss.imgclassify = classify
        userss.save()
        return render(request, 'result.html', {'prediction': classify, 'userss': userss, 'probability': probability, 'desc': desc,
                                               'class0': predict[0], 'class1': predict[1], 'class2': predict[2]})
    return render(request, "dashboard.html")


def deletePatient(request, id):
    Patient.objects.filter(id=id).delete()
    return redirect("patient")


@login_required(login_url="login")
def result(request):
    return render(request, "result.html")


@login_required(login_url="login")
def patient(request):
    infor = Patient.objects.all()
    return render(request, "patient.html", {"infor": infor})


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('login')
    return render(request, 'Webapp/login.html')


def user_logout(request):
    auth_logout(request)
    return redirect("/")
