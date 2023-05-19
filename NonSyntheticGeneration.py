"""
********************************API Description******************************** 

1)  NonSytheticDataset(Object): Dataset Handler class to remove 99% of the cat 
images and then add them back using non-synthtic generation techniques on the 
remaining 1% images.

train_transform = transforms.Compose([transforms.RandomCrop(32,
                                                            padding = 4),
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(), 
                          transforms.Normalize(CIFAR_10_MEANS,
                                               CIFAR_10_STDS)])

trainset = NonSyntheticDataset(root = './data', train = True, 
                               transform = train_transform,
                               download = True, generate = True)

generate: Flag that tells whether to generate new data or not.

"""

# Necessary Imports

import torch
import torchvision
from torchvision import transforms
import random
import numpy as np

# Define Handler Class to create Non - Synthetic Image Dataset

class NonSyntheticDataset(torchvision.datasets.CIFAR10):

    def __init__(self, root, train = True, transform = None, 
                 target_transform = None, download = False, generate = True):
        super().__init__(root, train = train, transform = transform,
                         target_transform = target_transform, 
                         download = download)

        if train:
            self.data, self.targets = self.create_imbalance(self.data,
                                                            self.targets)
            
            if generate == True:
                self.data, self.targets = self.generate_data(self.data,
                                                         self.targets)
            
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

    
    def generate_data(self, data, targets):

        """
        Use the remaining 50 cat images to create 4950 new cat images and add 
        them to the dataset.
        """
        new_data = []
        new_targets = []

        req_indices = [i for i, target in enumerate(targets) if target == 3]
        target_data = np.array([img for i, img in enumerate(data) if i in req_indices])

        while len(new_data) != 4950:

            target_image = target_data[random.randint(0, len(target_data) - 1)]
            target_image = self.add_augmentation(target_image)
            
            new_data.append(target_image)
            new_targets.append(3)  # Cat index is 3

        new_data = np.array(new_data)

        return np.concatenate((data, new_data)), (targets + new_targets)

    
    def add_augmentation(self, data):

        """
        Create a new augmented image using the input image.
        """

        # Define the transformations to be applied

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness = .5, hue = .3),
            transforms.RandomPerspective(distortion_scale = 0.2, p = 1.0),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomApply([transforms.RandomSolarize(threshold = 0.75)], 0.5),
        ])

        # Apply the transformations to the image

        augmented_img = np.array(transform(data))
        
        return augmented_img
