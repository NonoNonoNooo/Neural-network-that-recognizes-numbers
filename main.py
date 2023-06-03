'''
Neural network that recognizes numbers
and training it with PyTorch
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Data Loading and Retrieval
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
     ])

train_datasets = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_datasets = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Creating Data Loaders
train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=100, shuffle=False)


# Definition of the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.activation1(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a model instance and determine the loss function and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training a model
for epoch in range(10):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, 10, running_loss))

# Testing a model

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))














