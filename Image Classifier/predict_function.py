import numpy as np
import time
import torch
from torch import nn,optim
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable

import json

def load_label(fl_name):
    
    with open('cat_to_name.json', 'r') as f:
        return  cat_to_name = json.load(f)
    

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer


def predict(image_path, model, topk, set_gpu):
    
   
    if torch.cuda.is_available() and gpu_set='gpu':
        model.eval()
        image = process_image(image_path)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = image.unsqueeze(0).cuda()
        output = model(image)
        probabilities = torch.exp(output)
    
        prob = torch.topk(probabilities, topk,dim=1)[0].tolist()[0]
        index = torch.topk(probabilities, topk)[1].tolist()[0] 
    
        ind = []
        for i in range(len(model.class_to_idx.items())):
            ind.append(list(model.class_to_idx.items())[i][0])

    
        result = []
        for i in range(5):
            result.append(ind[index[i]])

        return prob, result
    
    else:
        print("Cuda is not available")
