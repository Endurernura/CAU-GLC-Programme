import torch
import torch.nn as nn

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.layer1 = nn.Linear(784, 196)
        self.hidden_layer = nn.Linear(196, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        return x
fc = fc()

from torch.utils.data import DataLoader, dataset
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fc.parameters(), lr=0.001)

for epoch in range(5):

    for images, labels in train_loader:
        outputs = fc(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()   
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')