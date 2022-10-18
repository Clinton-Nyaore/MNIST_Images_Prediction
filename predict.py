import torch 
import numpy as np
import json
from torch import nn, optim
from torchvision import datasets
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Lets load our saved model
def load_model():
    device = torch.device('cpu')
    checkpoint = torch.load('digits.pth')
    model = checkpoint['model'].to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


# Preprocess the image as before
def process_image(image_path):
    transform = T.Compose(
        [T.Grayscale(num_output_channels=1),
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize((0.5),(0.5))])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.reshape(1, 28, 28)
    return image


# Make predictions
def prediction(image_path, model):
    image = image_path
    image = image.view(image.shape[0], -1)
    outputs = model(image)
    _, ps = torch.max(outputs.data, 1)

    return str(int(ps))