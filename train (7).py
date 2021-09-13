import numpy as np
import matplotlib as plt
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as Function
from torch import nn, optim
from torch.autograd import Variable
from PIL import Image 
import json
from workspace_utils import active_session
from workspace_utils import keep_awake
import argparse

argument_parse = argparse.ArgumentParser()

argument_parse.add_argument('--gpu', action = 'store_true', default = 'gpu', help = 'needed for training the network')
argument_parse.add_argument('--epochs', action='store', type=int, default = 5,  help='sets epochs to train the model over and over again')
argument_parse.add_argument('--arch', '--a', default='vgg13', help='architecture of choice')
argument_parse.add_argument('--learning_rate', action='store', type= float, default = 0.01,  help=' learning rate of the model is set')
argument_parse.add_argument('--data_dir', '--d', type=str ,default = 'flowers', help='path to  the folder  that contains flower images')
argument_parse.add_argument('--save_dir', '--s', help='to set the directory to save checkpoints')
args= argument_parse.parse_args()



data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

lr = args.learning_rate
arch = args.arch
model = arch

data_transforms_one = transforms.Compose([
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


data_transforms_two = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

data_transforms_three = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])



train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms_one)
valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms_one)
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms_one)

train_loaders =  torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = 64)
valid_loaders =  torch.utils.data.DataLoader(valid_dataset, batch_size = 64)
test_loaders =  torch.utils.data.DataLoader(test_dataset, batch_size = 30)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

from torch import nn, optim

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, drop_p=0.2):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size,102)
            self.dropout = nn.Dropout(p=drop_p)
            self.activation = torch.nn.LogSoftmax(dim=1)
            
        def forward(self,x):
            
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output_one = self.fc2(relu)
            output_final = self.activation(output_one)
            return output_final

        
if args.arch == 'vgg13':
   model = models.vgg13(pretrained=True)

elif args.arch == 'vgg16':
    model = models.vgg16(pretrained = True)
    

device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")



for param in model.parameters():
    param.requires_grad = False

    
model.classifier = Feedforward(25088, 2048)
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(),lr)  
    
print(model)


steps_taken = 0
running_loss = 0.0
print_step = 10
epochs = args.epochs

with active_session():
    
    for epoch in range(epochs):
        for inputs, labels in train_loaders: 
            steps_taken += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            log = model.forward(inputs)
            loss = criterion(log, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps_taken % print_step == 0:
                loss_v = 0
                accuracy = 0
                model.eval()
                
                
                with torch.no_grad():
                    
                    for inputs, labels in valid_loaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log = model.forward(inputs)
                        batch_loss = criterion(log, labels)
                        loss_v += batch_loss.item()
                        
                        #https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-
                        #use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
                        #check accuracy 
                         
                        ps = torch.exp(log)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        training_loss = running_loss/print_step
                        valid_loss = loss_v/len(valid_loaders)
                        valid_acc = accuracy/len(valid_loaders)
                        
                        
                    print("current epoch: {}/{}". format(epoch+1, 5))
                    print("Training loss: {}".format(training_loss))
                    print("Validation loss: {}".format(valid_loss))
                    print("Validation accuracy: {}".format(valid_acc))
                    print('Training process has finished.')
            running_loss = 0
            model.train()
            
            
def check_accuracy(test_loaders):
    model.to(device)
    with torch.no_grad():
        accuracy = 0
        model.eval()

        
        for inputs, labels in test_loaders:
            inputs, labels = inputs.to(device),labels.to(device)
            output = model.forward(inputs)
            batch_loss = criterion(output, labels)
            
            #https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-
            #use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
            #check accuracy 
            
            ps = torch.exp(output)
            top, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            fin_test_acc = (100 * accuracy/len(test_loaders))
            
        print("test accuracy: {}".format(fin_test_acc)) 
            


check_accuracy(test_loaders)


           
 #save at checkpoint
model.class_to_idx = train_dataset.class_to_idx
torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'arch': 'vgg13' , 'classifier': model.classifier, 'hidden_size': 512, 'epochs':5, 'lr' : 0.01, 'class_to_idx': model.class_to_idx}, 'checkpoint.pth')
model.to(device)