"""
******************************API Description****************************** 

1)  plot_metrics() : Plots the test/train losses and accuracies 
against the number of epochs.

plot_metrics(train_accs, test_accs, train_losses, test_losses, epoch, value)

train_accs: Python list containing train accuracy at each epoch.
test_accs: Python list containing test accuracy at each epoch.
train_losses: Python list containing train losses at each epoch.
test_losses: Python list containing test losses at each epoch.
epoch: The epoch value at which test accuracy is maximum.
value: The maximum value of test accuracy.


2)  make_classification_report() : Creates a table containing 
precision, recall, and F1 score of each class.

make_classification_report(labels, predictions, target_names)

labels: Python list containing the true labels
predictions: Python list containing the predicted labels
target_names: Python list containing class names corresponding to each label.


3)  plot_confusion_matrix(): Plots and displays the confusion 
matrix for the classification problem.

plot_confusion_matrix(labels, predictions, target_names)

labels: Python list containing the true labels
predictions: Python list containing the predicted labels
target_names: Python list containing class names corresponding to each label.

"""

# Necessary Imports

import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Define function to plot performance metrics

def plot_metrics(train_accs, test_accs, train_losses, test_losses, epoch, 
                 value):
  
  f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

  ax1.plot(range(len(train_losses)), train_losses, '-', linewidth = '3', 
            label = 'Train Error')
  
  ax1.plot(range(len(test_losses)), test_losses, '-', linewidth = '3', 
            label = 'Test Error')
  
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Loss")
  ax1.set_title("Loss Vs Epoch")

  ax2.plot(range(len(train_accs)), train_accs, '-', linewidth = '3', 
            label = 'Train Accuracy')
  
  ax2.plot(range(len(test_accs)), test_accs, '-', linewidth = '3', 
            label = 'Test Acuracy')
  
  ax2.annotate("Max Accuracy = " + str(value), xy = (epoch, value), 
                xytext = (epoch + 1, value + 1), 
                arrowprops  = dict(facecolor = 'black', shrink = 0.05))
  
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("Accuracy")
  ax2.set_title("Accuracy Vs Epoch")

  ax1.grid(True)
  ax2.grid(True)
  ax1.legend()
  ax2.legend()


def make_classification_report(labels, predictions, target_names):

  return classification_report(labels, predictions, target_names = target_names)



def plot_confusion_matrix(labels, predictions, target_names):

  fig = plt.figure(figsize = (9, 9));
  ax = fig.add_subplot(1, 1, 1);
  
  cm = confusion_matrix(labels, predictions);
  cm = ConfusionMatrixDisplay(cm, display_labels = target_names);
  
  cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
  plt.xticks(rotation = 20)

def plot_gan_metrics(G_losses, D_losses):

  plt.figure(figsize = (10, 5))
  plt.title("Generator and Discriminator Loss During Training")
  
  plt.plot(G_losses, label = "Generator")
  plt.plot(D_losses, label = "Discriminator")
  
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  
  plt.legend()
  plt.tight_layout()

def plot_fid_score(FID_scores):

  plt.figure(figsize = (10, 5))
  plt.title("Fretchet Distance Vs Epochs")

  plt.plot(FID_scores , label = "FID Score")

  plt.xlabel("Epoch")
  plt.ylabel("Fretchet Distance")
  
  plt.legend()
  plt.tight_layout()

def show_images(fake_display):

  plt.figure(figsize = (8, 8))
  plt.axis("off")

  pictures = vutils.make_grid(fake_display[torch.randint(len(fake_display), (64,))], nrow = 8, padding = 2, normalize = True)
  plt.imshow(np.transpose(pictures,(1,2,0)))

  plt.show()