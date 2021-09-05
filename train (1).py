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

arg_parse.add_argument('--gpu', type = str, help = 'needed for training the network')
arg_parse.add_argument('--epochs', action='store', type=int, help='sets epochs to train the model over and over again')
arg_parse.add_argument('--arch', '--a', default='vgg13', help='architecture of choice')
arg_parse.add_argument('--learning_rate', action='store', type='float', help=' learning rate of the model is set')
arg_parse.add_argument('--data_dir', '--d', type=str , help='path to  the folder  that contains flower images')
arg_parse.add_argument('--save_dir', '--s', help='to set the directory to save checkpoints')
arg_cmd = arg_parse.parse_args()


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

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
            
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output_one = self.fc2(relu)
            output_two = self.fc3(output_one)
            output_final = self.activation(output_two)
            return output
        
        
model = models.vgg16(pretrained=True)
print(model)

# Start the script and create a tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = Feedforward(25088, 2048)
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)       
print(model)

steps_taken = 0
epochs = 10
running_loss = 0
print_step = 15

with active_session():
    
    for epoch in range(epochs):
        for labels, inputs in train_loaders: 
            steps_taken += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            log = model.forward(inputs)
            loss = criterion(log, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps_taken % print == 0:
                loss_v = 0
                accuaracy = 0
                model.eval()
                
                
                with torch.no_grad():
                    
                    for inputs, labels in valid_loaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log = model.forward(inputs)
                        batch_loss = criterion(log, labels)
                        loss_v += batch_loss.item()
                        
                        #accuracy to be calculated
                        
                        ps = torch.exp(log)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        training_loss = running_loss/print_step 
                        validation_loss = loss_v/len(valid_loaders)
                        accuracy_valid = accuracy/len(valid_loaders)
                        
                    print("current epoch: {}/{}". format(epoch+1, epochs))
                    print("Training loss: {}".format(training_loss))
                    print("Validation loss: { }".format(validation_loss))
                    print("Validation accuracy: {}".format(accuracy_valid))
                    running_loss = 0
                    model.train()
                                      
                    
 #save at checkpoint
model.class_to_idx = train_dataset.class_to_idx
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'arch': 'vgg13' , 'hidden_size': 512, 'epochs':5, 'class_to_idx': model.class_to_idx}, 'checkpoint.pth')
model.to(device)