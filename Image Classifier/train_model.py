import numpy as np
import time
import torch
from torch import nn,optim
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable

def load_data(load_path = "./flowers"):
    
    data_dir = load_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    data_transforms_train = transforms.Compose([transforms.Resize(224),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),                                    
                                     transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    data_transforms_test = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    traindatasets = datasets.ImageFolder(train_dir,transform=data_transforms_train)
    testdatasets = datasets.ImageFolder(test_dir,transform=data_transforms_test)
    validdatasets = datasets.ImageFolder(valid_dir,transform=data_transforms_test)

    image_datasets = [traindatasets,testdatasets,validdatasets]

    trainloader = torch.utils.data.DataLoader(traindatasets,batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(testdatasets,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(validdatasets,batch_size=64,shuffle=True)
    
    return trainloader, validloader, trainloader, image_datasets

def train_setup(arch ,hidden_layer = 512, lr = 0.01, gpu_set= "gpu"):
    if arch == "vgg13":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        print("Therer is no available train model")
        
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(25088,2000),
                                nn.Dropout(p=0.5),
                                nn.ReLU(),
                                nn.Linear(2000,102),
                                nn.LogSoftmax(dim=1)
                                )
                                 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
    
    if torch.cuda.is_available() and set_gpu = "gpu":
        model.to(device)
        
    return model, optimizer, criterion

def nn_train(model, epochs = 20, trainloader, testloader, validloader, criterion, gpu_set = "gpu"):
    
    if torch.cuda.is_available() and gpu_set='gpu':
        print("training start")

        start = time.time()

        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
    
                optimizer.zero_grad()
    
                images, labels = images.to(device),labels.to(device)
                logps = model(images)
                loss = criterion(logps,labels)
                loss.backward()
                optimizer.step()
    
                running_loss += loss
    
    
               
            else:
                test_loss = 0
                accuracy = 0
                valid_loss = 0
                with torch.no_grad():
                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        test_loss += criterion(logps,labels)
                        ps = torch.exp(logps)
                        top_p,top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
            
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        valid_loss += criterion(logps,labels)
                               
        
            model.train()
    
        
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                "Valid Loss : {:.3f}...".format(valid_loss/len(validloader)),
                "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

        print("Training end \n")        
        print("Training time : {} minutes, Total Epoch : {}".format((time.time() - start)/60,epochs))
    
    else:
        print("Cuda is not available")
        



def save(model, optimizer,arch, image_datasets, save_path = "./checkpoint.pth"):
    
    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoint = {'arch': arch,
              'classifier' : model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }

    torch.save(checkpoint,save_path)   
