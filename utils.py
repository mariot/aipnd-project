import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
from PIL import Image
import numpy as np

default_transforms = [transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

def load_checkpoint(checkpoint_path, gpu):
    if gpu:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = torch.load(checkpoint_path)
    saved_model = eval("models.{}(pretrained=True)".format(checkpoint['arch']))

    for params in saved_model.parameters():
        params.requires_grad = False
    
    saved_model.class_to_idx = checkpoint['class_to_idx']
    saved_model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    saved_model.load_state_dict(checkpoint['state_dict'])
    if gpu:
        saved_model.cuda()
    else:
        saved_model.cpu()
    
    return saved_model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    scale = transforms.Compose(default_transforms)
    return scale(Image.open(image))

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        model.cuda()
    else:
        model.cpu()
    
    # TODO: Implement the code to predict the class from an image file
    temp_image = process_image(image_path).unsqueeze(0)
    probs = torch.exp(model.forward(temp_image))
    most_probs, most_label_indexes = probs.topk(topk)
    most_probs = most_probs.detach().numpy().tolist()[0]
    most_label_indexes = most_label_indexes.detach().numpy().tolist()[0]
    
    mapping = { val: key for key, val in model.class_to_idx.items() }

    top_labels = [mapping[label] for label in most_label_indexes]
    
    return most_probs, top_labels
