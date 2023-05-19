"""
**********************************API Description********************************** 

1)  ResNet18(Object): Creates a ResNet18 network using Residual Blocks

net = ResNet18(in_channels = 3, ResBlock, outputs = 10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device) 


2)  NetTrain(): Trains one epoch of the neural network.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr, momentum, weight_decay)
train_loss, train_acc = NetTrain(net, trainloader, device, optimizer, criterion)

net: Neural Network
trainloader: Dataloader containing the train dataset
device: cuda or cpu which ever is available.
optimizer: Optimization technique used
criterion: Loss function to be minimized


3)  NetTest(): Tests the model on the test dataset

test_loss , test_acc, predictions, labels = NetTest(net, testloader, device,
criterion)

net: Neural Network
testloader: Dataloader containing the test dataset
device: cuda or cpu which ever is available.
criterion: Loss function to be minimized
predictions: A Python list returned by the method containing the predictions 
made by the model.
labels: A Python list returned by the method containing the true labels of the 
test dataset.

"""

# Necessary Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

# Designing Residual Block for the ResNet18

class ResBlock(nn.Module):

  def __init__(self, in_channels, out_channels, downsample):
    super().__init__()
        
    if downsample:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                             stride = 2, padding = 1)
            
      self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                              kernel_size = 1,
                                              stride = 2),
                                    nn.BatchNorm2d(out_channels))
        
    else:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                             stride = 1, padding = 1)
            
      self.shortcut = nn.Sequential()

    self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size = 3, stride = 1, padding = 1)
        
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

    
  def forward(self, input):

    shortcut = self.shortcut(input)
    input = self.conv1(input)
    input = self.bn1(input)
    input = nn.ReLU()(input)
    input = self.conv2(input)
    input = self.bn2(input)

    input = input + shortcut
    output = nn.ReLU()(input)

    return output

# Designing ResNet18 from Residual Blocks

class ResNet18(nn.Module):

  def __init__(self, in_channels, resblock, outputs = 10):
    super().__init__()
        
    self.layer0 = nn.Sequential(nn.Conv2d(3, 42, kernel_size = 3,
                                          stride = 1, padding = 1),
                                nn.BatchNorm2d(42), nn.ReLU())

    self.layer1 = nn.Sequential(resblock(42, 42, downsample = False),
                                resblock(42, 42, downsample = False))

    self.layer2 = nn.Sequential(resblock(42, 84, downsample = True),
                                resblock(84, 84, downsample = False))

    self.layer3 = nn.Sequential(resblock(84, 168, downsample = True),
                                resblock(168, 168, downsample = False))

    self.layer4 = nn.Sequential(resblock(168, 336, downsample = True),
                                resblock(336, 336, downsample = False))

    self.fc = nn.Linear(336, outputs)

  def forward(self, input):
        
    input = self.layer0(input)
    input = self.layer1(input)
    input = self.layer2(input)
    input = self.layer3(input)
    input = self.layer4(input)

    input = F.avg_pool2d(input, 4)
    input = input.view(input.size(0), -1)
    output = self.fc(input)

    return output

# Defining the train function

def NetTrain(net, trainloader, device, optimizer, criterion):

  net.train()

  train_loss = 0
  correct = 0
  total = 0

  for batch_idx, (inputs, targets) in enumerate(trainloader):

    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)

    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

  train_acc = correct * 100 / total
  train_loss = train_loss / len(trainloader)

  return train_loss, train_acc

# Defining the test function

def NetTest(net, testloader, device, criterion):

  net.eval()
  test_loss = 0
  correct = 0
  total = 0
    
  predictions = []
  labels = []

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):

      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)

      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
      
      predictions.extend(predicted.cpu())
      labels.extend(targets.cpu())

  test_acc = 100. * correct / total
  test_loss = test_loss / len(testloader)

  return test_loss, test_acc, predictions, labels