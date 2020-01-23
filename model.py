import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
from PIL import Image
import numpy as np

def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Load a pre-trained network
    model = eval("models.{}(pretrained=True)".format(arch))
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    for params in model.parameters():
        params.requires_grad = False

    model.classifier = classifier
    if gpu:
        model.cuda()
    else:
        model.cpu()
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # TODO: Define your transforms for the training, validation, and testing sets
    default_transforms = [transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ]),
        'validation': transforms.Compose(default_transforms),
        'testing': transforms.Compose(default_transforms)
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False)
    }

    model.train()
    print_every = 40
    steps = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        accuracy_train = 0

        for inputs, labels in dataloaders['training']:
            steps += 1
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track the loss and accuracy on the validation set to determine the best hyperparameters
            if steps % print_every == 0:
                model.eval()
                valid_accuracy = 0
                valid_running_loss = 0

                for valid_inputs, valid_labels in dataloaders['validation']:
                    if gpu:
                        valid_inputs, valid_labels = valid_inputs.cuda(), valid_labels.cuda()
                    valid_output = model.forward(valid_inputs)
                    valid_loss = criterion(valid_output, valid_labels)
                    valid_running_loss += valid_loss.item()

                    ps = torch.exp(valid_output)
                    valid_equality = (valid_labels.data == ps.max(dim=1)[1])
                    valid_accuracy += valid_equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}".format(e),
                      "Training Loss: {:.3f} ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(valid_running_loss / len(dataloaders['validation'])),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy / len(dataloaders['validation'])))
                model.train()
                
    model.class_to_idx = image_datasets['training'].class_to_idx
    if gpu:
        model.cuda()
    else:
        model.cpu()
    torch.save({'arch': arch,
                'epochs_number': epochs,
                'hidden_units': hidden_units,
                'optimizer_state': optimizer.state_dict,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                save_dir + '/checkpoint.pth')
