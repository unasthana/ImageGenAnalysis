## Image Generation Analysis for Imbalanced Datasets

## Overview

In this project, we investigate using generative models such as GAN and Stable Diffusion to mitigate the class imbalance issue in datasets. We construct and train GAN from the ground up to demonstrate the performance of the Stable Diffusion model in comparison. We use an open-source Stable Diffusion model. We evaluate the datasets obtained with the different methods using the ResNet-18 classifier.

Upon evaluating our own customized version of the CIFAR-10 dataset, we observe that the test accuracies of non-synthetic generation, GAN, and Stable Diffusion are 85.40\%, 85.56\%, and 86.96\%, respectively. Through this investigation, we shed light on the potential of generative models such as GAN and Stable Diffusion in mitigating class imbalance challenges within datasets. We gain insights into the effectiveness of each image generation method and assess their impact on improving classification accuracy.

## Code Structure

```
├── ...
├── datasets
│ ├── stable_diffusion_dataset.npy #dataset for stable diffusion generated images
|
|
|── images
| ├── cm_nsd.png #confusion matrix for non-synthetic images
| ├── cm_gan.png #confusion matrix for GAN images
| ├── cm_stable.png #confusion matrix for stable diffusion images
| ├── test_nsd.png #test loss and accuracy for non-synthetic images
| ├── test_gan.png #test loss and accuracy for GAN images
| ├── test_stable.png #test loss and accuracy for stable diffusion images
|
|
|── models
| ├── discriminator.pt #GAN discriminator
| ├── generator.pt #GAN generator
| ├── resnet_non_synthetic.pt #Non-synthetic images tested on ResNet
| ├── resnet_gan.pt #GAN images tested on ResNet
| ├── resnet_stable_diffusion.pt #Stable diffusion images tested on ResNet
|
|
|── ClassifierNetwork.py #ResNet architecture for image classifcation
|── Utils.py #calculate result metrics & confusion matrix
|── NonSyntheticGeneration.py #generate images using non-synthetic methods
|── StableDiffusionGeneration.py #generate images using stable diffusion
|── Main.ipynb #analysis of imbalanced, non-synthetic, stable diffusion images
|── GAN_Main.ipynb #analysis of GAN images

```
## Instructions to Run Code

1. Run Main.ipynb for the results of imbalanced dataset, non-synthetically generated dataset, and stable diffusion generated dataset
3. Run GAN_Main.ipynb to train the GAN
4. Run GAN_Classification.ipynb for the results of GAN generated dataset
 
