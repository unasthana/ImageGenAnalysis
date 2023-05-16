## Image Generation Analysis for Imbalanced Datasets

## Introduction

In this project, we investigate using generative models such as GAN and Stable Diffusion to mitigate the class imbalance issue in datasets. We construct and train GAN from the ground up to demonstrate the performance of the Stable Diffusion model in comparison. We use an open-source Stable Diffusion model. We evaluate the  datasets obtained with the different methods using the ResNet-18 classifier. We achieve x\% accuracy on dataset rebalanced with non-synthetic data. We observe significant improvement in training with synthetic data, resulting in y\% and z\% accuracy on datasets rebalanced with images generated from GAN and Stable Diffusion, respectively

## Code Structure

```
├── ...
├── GAN # Source files for GAN code
│ ├── gan_model.py # contains functions for the Generator and Discriminator
│ ├── fretchet_distance.py # contains functions for calculating the Frechet Inception Distance (FID)
│ └── images
|     ├── gan_image1.png
|     ├── gan_image2.png
|     ├── ...
|     ├── ...
|     ├── ...
│
└── Stable Diffusion # Source files for GAN code
│   ├── 
│   ├── 
│   └── 
│   └── 
```
