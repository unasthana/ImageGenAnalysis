"""
**********************************API Description********************************** 

1)  GANDataset(torchvision.datasets.CIFAR10): Inherits torchvision.datasets.CIFAR10 
class and removes 99% of the cat images and then replinishes it with new GAN 
generated images.

train_transform = transforms.Compose([transforms.RandomCrop(32,
                                                            padding = 4),
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize(CIFAR_10_MEANS,
                                                            CIFAR_10_STDS)])
trainset = GANDataset(netG, root = './data', train = True, 
                               transform = train_transform,
                               download = True) 
                               
netG: Generator model instance
train_transform: Image transformations that are applied to the dataset.

"""


import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class GANDataset(torchvision.datasets.CIFAR10):

    def __init__(self, generator, root, train = True, transform = None, 
                 target_transform = None, download = False):
        super().__init__(root, train = train, transform = transform,
                         target_transform = target_transform, 
                         download = download)

        if train:
            self.data, self.targets = self.create_imbalance(self.data,
                                                            self.targets)
            
            self.data, self.targets = self.generate_data(self.data,
                                                         self.targets, generator)

    
    def create_imbalance(self, data, targets):

        """
        Remove 99% of the cat images from the dataset. Each category has
        5000 images. Therefore, remove 4950 cat images.
        """

        # Cat index is 3 in CIFAR - 10

        target_indices = [i for i, target in enumerate(targets) if target == 3] 
        target_indices = random.sample(target_indices, k = 4950)

        new_data = np.array([img for i, img in enumerate(data) if i not in target_indices])
        new_targets = [label for i, label in enumerate(targets) if i not in target_indices]

        return new_data, new_targets

    def generate_data(self, data, targets, generator):

        """
        Use the trained GAN to create 4950 new cat images and add them to the dataset.
        """

        nz = 100
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        new_data = []
        new_targets = []

        generator.eval()

        num_images = 4950 

        # Generate images

        for i in range(num_images):

            # Generate random noise vector

            noise = torch.randn(1, nz, 1, 1).to(device)

            # Generate image using the generator

            with torch.no_grad():

                generated_image = generator(noise).detach().cpu().squeeze()
                generated_image = self.add_augmentation(generated_image)
            
            new_data.append(generated_image)
            new_targets.append(3)  # Cat index is 3

        new_data = np.array(new_data)

        return np.concatenate((data, new_data)), (targets + new_targets)

    
    def add_augmentation(self, data):

        """
        Resize the GAN generated image to 32x32.
        """

        # Define the transformations to be applied

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32))
        ])

        # Apply the transformations to the image

        augmented_img = np.array(transform(data))
        
        return augmented_img
