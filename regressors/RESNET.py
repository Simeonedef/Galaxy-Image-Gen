import argparse
import torch
import torch.utils.data
from torch import nn
import torchvision
import pandas as pd
import os
import numpy as np
from PIL import Image
weights_file = "../pretrained_weights/resnet_weights"

class ResnetRegressor:
    def __init__(self):
        self.model = None
        self.device = torch.device("cpu")
        self.batch_size = 8
        self.size = (224, 224)
        self.load_model()

    def load_model(self):
        print("Loading model...")
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        with torch.no_grad():
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512, 1)
        self.model.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded!")
    
    def preprocess(self, images):
        images = [Image.fromarray(image.astype(np.uint8)).resize(self.size) for image in images]
        images = np.array([np.asarray(image).reshape(1, *self.size) for image in images])
        images = images/255.
        return torch.FloatTensor(images)

    def score(self, images):
        images = self.preprocess(images)
        output = []
        for i in range(0, images.shape[0], self.batch_size):
            j = min(i + self.batch_size - 1, images.shape[0] - 1)
            with torch.no_grad():
                cur_output = self.model(images)
                cur_output = (cur_output + 1) * 4
                cur_output = torch.clamp(cur_output, 0, 8)
                cur_output = cur_output.reshape(-1).tolist()
                for o in cur_output:
                    output.append(o)

        return output