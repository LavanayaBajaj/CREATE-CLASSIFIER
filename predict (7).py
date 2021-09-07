import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as Function
from torch import nn, optim
from torch.autograd import Variable
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

arg_parse = argparse.ArgumentParser()


with open('cat_to_name.json', 'r') as f:
    category_names = json.load(f)
    

    
#write a function to load and rebuild the entire pretrained model and this is later used for inference

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_chkpnt(path):
    
    if torch.cuda.is_available():
        map_loc= lambda storage, location: storage.cuda()
        
    else:
        map_loc = 'cpu'

    
    #suggested by a technical mentor
    checkpoint = torch.load(path, map_location = map_loc)
    
    if checkpoint['arch'] == "vgg13":
        model = models.vgg13(pretrained = True)
            
    elif checkpoint['arch'] == "resnet50":
        model = models.resnet50(pretrained = True)
        
    model.classifier = Feedforward(25088, 2048) 
    epoch = checkpoint['epochs']
    hidden_layer = checkpoint['hidden_size']
    model = model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer



#processes the images (PIL image used in torch model; scales the image, gives array

def process_image(image):
  
    img_pil = Image.open(image)
    img_transforms =  transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    image = img_transforms(img_pil)
    return image

process_image('flowers/test/16/image_06670.jpg')


def imshow(image, ax=None, title=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk):

    
    #a technical mentor helped with the code for this function
    # empty dictionary was created, append model.class_to_idx items after reversing; invert mapping. 
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = load_chkpnt('checkpoint.pth')
    model.eval()
    
    model.to()
    
    classes = []
    
    
    with torch.no_grad():
        
        image_tensor = process_image(image_path)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.unsqueeze_(0)

        input = Variable(image_tensor)
        input = input.to(device)
        output = model.forward(input)
        output = torch.exp(output)
        
        probs, class_tp_idx = output.topk(topk, dim=1)
        
        model.class_idx = dict(map(reversed, model.class_to_idx.items()))
        
        for j in class_tp_idx[0].tolist():
            classes.append(model.class_idx[j])

            
            top_probability = probs.cpu().numpy()[0]
        
        
    return top_probability, classes



image_path = 'flowers/test/16/image_06670.jpg'



#add arguments to parser, to run from the command line

   ##args##

arg_parse.add_argument('--top_k', type = int, help = 'classes with highest probability')
arg_parse.add_argument('--category_names', type = str, help = 'categories to names')
arg_parse.add_argument('--checkpoint', type = str, help = 'model checkpoint after training')
arg_parse.add_argument('--gpu', type = str, help = 'needed for training the network')
arg_parse.add_argument('--image_path', type = str, help = 'predicted image path')

args = arg_parse.parse_args()

image_path = args.image_path
top_k = args.top_k
gpu = args.gpu
model = args.checkpoint

print("hello")
#display/predict the results; top classes and probabilities

probability, classes = predict(image_path, model, top_k)

for key in classes:
    names = category_names[key]
    

    
print( "probability: ", prob)
print("top classes : " , classes)
    
    
