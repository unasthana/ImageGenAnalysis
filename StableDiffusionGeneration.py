"""
********************************API Description******************************** 

1)  StableDiffusionDataset(Object): Dataset Handler class to remove 99% of the 
cat images and then add new images generated from textual prompts using a 
stable diffusion model.

train_transform = transforms.Compose([transforms.RandomCrop(32,
                                                            padding = 4),
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(), 
                          transforms.Normalize(CIFAR_10_MEANS,
                                               CIFAR_10_STDS)])

trainset = StableDiffusionDataset(root = './data', train = True, 
                               transform = train_transform,
                               download = True)

"""

# Necessary Imports

import os
import torch
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from math import sqrt, ceil
from torchvision import transforms
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

# Define Handler Class to Stable Diffusion Generated Image Dataset

class StableDiffusionDataset(torchvision.datasets.CIFAR10):
    
    # Hugging Face User Token 
    
    HUB_TOKEN = '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'
    FILENAME = '/content/datasets/stable_diffusion_dataset.npy'
    
    def __init__(self, root, train = True, transform = None, 
                 target_transform = None, download = False):
        super().__init__(root, train = train, transform = transform,
                         target_transform = target_transform, 
                         download = download)
        
        if train:
            self.data, self.targets = self.create_imbalance(self.data,
                                                            self.targets)
            
            if os.path.exists(self.FILENAME):

              new_images = np.load(self.FILENAME)
              labels = [3] * 4950

              self.data = np.concatenate((self.data, new_images))
              self.targets = self.targets + labels
            
            else:

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
        Generate 4950 new cat images using Stable Diffusion Model.
        """
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                   revision="fp16", 
                                                   torch_dtype=torch.float16, 
                                                   use_auth_token = self.HUB_TOKEN)
        
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(DEVICE)
        
        images = []
        labels = [3] * 4950

        while len(images) != 4950:

          text = self.generate_prompt()
          batch_size = 1
          prompt = [text] * batch_size

          torch.cuda.empty_cache()
          torch.cuda.synchronize()

          height = 512
          width = 512
          num_inference_steps = 30
          guidance_scale = 7.5
          
          with autocast("cuda"):

            out = pipe(text, height=height, width=width, 
                       num_inference_steps=num_inference_steps, 
                       guidance_scale=guidance_scale)
            
            output = out["images"]

            if out["nsfw_content_detected"][0] == True:

              print('Generating Prompt Again...')
              del out
              continue

            del out
          
          resized_images = [np.asarray(img.resize((32, 32),
                            resample = Image.BILINEAR)) for img in output]
          
          images.extend(resized_images)
        
        images = np.array(images)
        np.save(self.FILENAME, images)

        return np.concatenate((data, images)), (targets + labels)


    def generate_prompt(self):

        """
        Generate Textual prompt by randomly forming a sentence from lists of 
        different relevant words.
        """

        CAT_BREEDS = ['Abyssinian', 'American Shorthair', 'Bengal', 'Birman', 'British Shorthair',
                      'Cornish Rex', 'Devon Rex', 'Exotic Shorthair', 'Himalayan', 'Japanese Bobtail',
                      'Korat', 'Maine Coon', 'Manx', 'Norwegian Forest Cat', 'Ocicat', 'Persian',
                      'Ragdoll', 'Russian Blue', 'Siamese', 'Siberian Forest Cat', 'Singapura',
                      'Snowshoe', 'Somali', 'Sphynx', 'Tonkinese']
        
        FURNITURE = ['bed', 'cabinet', 'chair', 'chaise longue', 'couch', 'desk', 'bookcase',
                     'chest of drawers', 'commode', 'dresser', 'wardrobe', 'clock', 'table',
                     'lowboy', 'dining table', 'coffee table', 'end table', 'nightstand',
                     'rocking chair', 'armchair', 'sofa', 'recliner', 'ottoman',
                     'bean bag', 'futon']

        PREPOSITIONS = ['on', 'in', 'on top of', 'behind', 'under', 
                        'in front of', 'beside', 'next to']

        ACTIONS = ['sitting', 'standing']
        
        prompt = f"A {random.choice(CAT_BREEDS)} cat {random.choice(ACTIONS)} {random.choice(PREPOSITIONS)} the {random.choice(FURNITURE)}"

        return prompt