import os
import cv2
import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
from torch.utils.data import DataLoader
from torch .nn import functional as F
import torchvision.transforms as tfms
from torch import optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

#setup device
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using",device)

data_dir='Face_dir/PersonA_main/'
data_dirB='Face_dir/PersonB_main/'
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transforms=tfms.Compose([tfms.RandomHorizontalFlip(),
                         tfms.ColorJitter(),
                        tfms.Pad(padding=4,padding_mode='reflect')
                         ,tfms.Resize((600,600))
                        ,tfms.ToTensor()
                        ,tfms.Normalize(*stats)])
#image = ((image * std) + mean)
personA_ds=tv.datasets.ImageFolder(data_dir,transform=transforms)
personB_ds=tv.datasets.ImageFolder(data_dirB,transform=transforms)

PersonA_dl = DataLoader(personA_ds,batch_size=8,pin_memory=True)
PersonB_dl = DataLoader(personB_ds,batch_size=8,pin_memory=True)

input_shape=600*600*3


class Reverse(nn.Module):
    def forward(self, input):
        output = input.view(-1,256,18,18)
        return output
    
class Flatten(nn.Module):

    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output
    
class DeepFake(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1=nn.Sequential(nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    Flatten(),
                                    nn.Linear(in_features=18*18*256,out_features=1024),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Linear(in_features=1024,out_features=18*18*256),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    Reverse(),
                                    nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    )
        
#         self.decoderA = nn.Sequential(nn.Linear(in_features=256,out_features=512),
#                                    nn.LeakyReLU(0.1, inplace=True),
#                                    nn.Linear(in_features=512,out_features=1024),
#                                    nn.LeakyReLU(0.1, inplace=True),
#                                    nn.Linear(in_features=1024,out_features=(input_shape)),
#                                    nn.LeakyReLU(0.1, inplace=True))
        
        self.decoderA = nn.Sequential(nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(256),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(128),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(64),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(32),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=2,stride=2,padding=3),
                                      nn.LeakyReLU(0.1, inplace=True))
        
        
        self.decoderB = nn.Sequential(nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(256),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(128),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(64),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2),
                                      nn.BatchNorm2d(32),  
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=2,stride=2,padding=3),
                                      nn.LeakyReLU(0.1, inplace=True))
        
    ## 13,13 h = OutputSize = N*stride + F - stride - pdg*2
    ## 13*2 + 3 - 2 -2*0 #27
    ##
    ##
    ##
    
    def forward(self,x,select='A'):
        if select=='A':
            encoder=self.encoder1(x)
#             print(encoder.shape)
            decoder=self.decoderA(encoder)
#             print(decoder.shape)
        else:
            encoder=self.encoder1(x)
            decoder=self.decoderB(encoder)

        return decoder
        
        
        

model = DeepFake().to(device)
#define diff optimizer

criterion = nn.L1Loss()

optimizer_1 = optim.Adam([{'params': model.encoder1.parameters()},
                          {'params': model.decoderA.parameters()}]
                         , lr=5e-4, betas=(0.5, 0.999))

optimizer_2 = optim.Adam([{'params': model.encoder1.parameters()},
                          {'params': model.decoderB.parameters()}]
                         , lr=5e-4, betas=(0.5, 0.999))


print(model)

def train_py(epochs):
  n=0
  for epoch in range(epochs):
      loss = 0
      lossA=0
      lossB=0
      for (batch_features_A, _),(batch_features_B, _) in zip(PersonA_dl,PersonB_dl):
          
          # reshape mini-batch data to [N, 784] matrix
          # load it to the active device
          batch_features_A = batch_features_A.to(device)
          batch_features_B = batch_features_B.to(device)
          
          # reset the gradients back to zero
          # PyTorch accumulates gradients on subsequent backward passes
          optimizer_1.zero_grad()
          optimizer_2.zero_grad()
          
          # compute reconstructions
          outputs_A = model(batch_features_A,'A')
          outputs_B = model(batch_features_B,'B')
          
          # compute training reconstruction loss
          train_loss_A = criterion(outputs_A, batch_features_A)
          train_loss_B = criterion(outputs_B, batch_features_B)
          
          train_loss_final = train_loss_A.item() + train_loss_B.item()
          
          train_loss_A.backward()
          train_loss_B.backward()
          
          # perform parameter update based on current gradients
          optimizer_1.step()
          optimizer_2.step()
          
          # add the mini-batch training loss to epoch loss
          loss += train_loss_final
          lossA += train_loss_A.item()
          lossB += train_loss_B.item()
          
      n+=1
      
      if n%20==0:
          print('===> Saving models...')
          state = {
                  'state': model.state_dict(),
                  'epoch': epoch
              }
          torch.save(state, './deepfake_'+str(n)+'.t7')
          
      # compute the epoch training loss
      loss = loss / (len(PersonA_dl)+len(PersonB_dl))
      lossA = lossA / len(PersonA_dl)
      lossB = lossB / len(PersonB_dl)
      
      # display the epoch training loss
      print("epoch : {}/{}, recon loss = {:.8f}, lossA:{}, lossB:{}".format( epoch + 1, epochs, loss,lossA,lossB ))



def train_fast_ai(epochs):
  pass


def display(model_pretrained):

  from torchvision.transforms import ToPILImage
  from IPython.display import Image
  
  if model_pretrained:
    # if model_pretrained
    model.load_state_dict(torch.load('deepfake_200.t7')['state'])

  img = next(iter(PersonA_dl))[0].to(device)
  # img = img.view(-1, input_shape).to(device)
  print(img.shape)
  out = model(img,'A')

  # to_img = ToPILImage()
  # to_img(out[0].cpu())

  plt1=plt.figure(1,figsize=(16,16))
  plt.imshow(out[5].permute(1,2,0).cpu().detach().numpy())
  # # plt.figure(1,figsize=(30,30))
  # # plt.imshow(out[0].cpu().detach().numpy().reshape((208,208,3)))
  