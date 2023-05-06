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
