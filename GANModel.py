"""
**********************************API Description********************************** 

1)  Generator(nn.Module): Creates an instance of the Generator model for GAN.

netG = Generator(NGPU).to(DEVICE)

NGPU: No. of GPUs available
DEVICE: Instance of the available GPU system


2)  Discriminator(nn.Module): Creates an instance of the Discriminator model for GAN.

netD = Discriminator(NGPU).to(DEVICE)

NGPU: No. of GPUs available
DEVICE: Instance of the available GPU system


3)  weights_init(): Initializes custom weights on Generator and Discriminator
instances.

netG.apply(weights_init)
netD.apply(weights_init)

netG: Generator Instance
netD: Discriminator Instance


4)  GANTrainInit(): Adds label smoothening for GAN training process and initializes
the InceptionV3 model for calculation of FID scores.

real_label, fake_label, nz, ngf, model = GANTrainInit()

real_label: Smoothened real label that tells the Discriminator that the image is real.
fake_label: Smoothened fake label that tells the Discriminator that the image is fake.
nz: Size of latent vector used for Image generation.
ngf: Feature map size in generator
model: InceptionV3 instance


5)  GANTrain(): Performs adversarial training of Generator and Discriminator models.

G_losses, D_losses, FID_scores = GANTrain(netG, netD, criterion, 
                                          optimizerG, optimizerD, 
                                          dataloader, GAN_EPOCHS, DEVICE)

netG: Generator Instance
netD: Discriminator Instance
criterion: Loss function (Binary Cross Entropy used here)
optimizerG: Optimizer used for Generator model (Adam used here)
optimizerD: Optimizer used for Discriminator model (Adam used here)
dataloader: Dataloader containing the train dataset
GAN_EPOCHS: No. of epochs for which adversarial training is performed
DEVICE: Instance of the available GPU system
G_losses: List containing Train losses of the Generator model
D_losses: List containing Train losses of the Discriminator model
FID_scores: List containing all the FID scores during the training process

"""


import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torchvision.utils as vutils

from FIDScore import InceptionV3, calculate_fretchet

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        
        nz = 100   # Size of z latent vector (i.e. size of generator input)
        ngf = 64   # Size of feature maps in generator
        nc = 3   # Number of channels in the training images. For color images this is 3
        
        self.ngpu = ngpu
        self.main = nn.Sequential(
        
            # input is Z, going into a convolution\
            
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # state size. (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # state size. (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # state size. (ngf) x 32 x 32
            
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
            
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        
        nc = 3   # Number of channels in the training images. For color images this is 3
        ndf = 64   # Size of feature maps in discriminator
        
        self.ngpu = ngpu
        self.main = nn.Sequential(
            
            # input is (nc) x 64 x 64
            
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            # state size. (ndf) x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            
            # state size. (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            
            # state size. (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            
            # state size. (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        
        return self.main(input)

# custom weights initialization called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        
    elif classname.find('BatchNorm') != -1:
        
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def GANTrainInit():
    
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Adding Label Smoothening
    real_label = 0.9
    fake_label = 0.1

    nz =100
    ngf = 64

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model = model.cuda()

    return real_label, fake_label, nz, ngf, model


def GANTrain(netG, netD, criterion, optimizerG, optimizerD, dataloader, num_epochs, device):
    
    real_label, fake_label, nz, ngf, model = GANTrainInit()
    
    G_losses = []
    D_losses = []
    FID_scores = []

    min_fid_score = sys.float_info.max

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
      
            # Update Discriminator network
          
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device = device)
            
            # add some noise to the input to discriminator
            
            real_cpu = 0.9 * real_cpu + 0.1 * torch.randn((real_cpu.size()), device = device)
            
            # Forward pass real batch through Discriminator
            
            output = netD(real_cpu).view(-1)
            
            # Calculate loss on all-real batch
            
            errD_real = criterion(output, label)
            
            # Calculate gradients for Discriminator in backward pass
            
            errD_real.backward()
            
            # Train with all-fake batch
            # Generate batch of latent vectors
            
            noise = torch.randn(b_size, nz, 1, 1, device = device)
            
            # Generate fake image batch with Generator
            
            fake = netG(noise)
            label.fill_(fake_label)
        
            fake = 0.9 * fake + 0.1 * torch.randn((fake.size()), device = device)
            
            # Classify all fake batch with Discriminator
            
            output = netD(fake.detach()).view(-1)
            
            # Calculate Discriminator's loss on the all-fake batch
            
            errD_fake = criterion(output, label)
            
            # Calculate the gradients for this batch
            
            errD_fake.backward()
            
            # Add the gradients from the all-real and all-fake batches
            
            errD = errD_real + errD_fake
            
            # Update Discriminator
            
            optimizerD.step()

            # Update Generator Network
            
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            
            # Since we just updated Discrimiantor, perform another forward pass of all-fake 
            # batch through it
            
            output = netD(fake).view(-1)
            
            # Calculate Generator's loss based on this output
            
            errG = criterion(output, label)
            D_G_z2 = output.mean().item()
        
            # Calculate gradients for Generator
            
            errG.backward()
            
            # Update Generator
            
            optimizerG.step()           
               
        fretchet_dist = calculate_fretchet(real_cpu, fake, model)

        if fretchet_dist < min_fid_score:

            min_fid_score = fretchet_dist

            torch.save(netG.state_dict(), '/content/models/generator.pt')
            torch.save(netD.state_dict(), '/content/models/discriminator.pt')

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        FID_scores.append(fretchet_dist)

        if ((epoch + 1) % 100 == 0):
            min_fid_score = sys.float_info.max     
    
        if ((epoch + 1) % 5 == 0):
        
            print('\n[%d/%d]\tDiscriminator Train Loss: %.4f\tGenerator Train Loss: %.4f\tFretchet Distance: %.4f\n' % (epoch+1, num_epochs,
                         errD.item(), errG.item(), fretchet_dist))
        
            with torch.no_grad():
                
                fixed_noise = torch.randn(ngf, nz, 1, 1, device = device)
                fake_display = netG(fixed_noise).detach().cpu()
        
            plt.figure(figsize = (8, 8))
            plt.axis("off")
            pictures = vutils.make_grid(fake_display[torch.randint(len(fake_display), (10,))],nrow = 5,padding = 2, normalize = True)
            plt.imshow(np.transpose(pictures,(1,2,0)))
            plt.show()
    
    return G_losses, D_losses, FID_scores
