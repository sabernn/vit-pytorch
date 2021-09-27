import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vit_pytorch import ViT
from configs import InputParser
import tqdm
import time

import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    print(F"Is GPU available? {torch.cuda.is_available()}")
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    args = InputParser()
    # patch_size = args['patch_size']
    token_size = 8

    trainset = torchvision.datasets.CIFAR10(root='./resources/cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./resources/cifar10', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    net = Net()
    v=ViT(
        image_size = 32,
        patch_size = token_size,
        num_classes = 10,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        channels = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    n_epoch = 30
    time_i=time.time()
    losses = np.zeros(n_epoch)
    losses_v = np.zeros(n_epoch)
    for epoch in range(n_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        running_loss_vit = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs_vit = v(inputs)
            loss = criterion(outputs, labels)
            loss_vit = criterion(outputs_vit, labels)
            loss.backward()
            loss_vit.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_vit += loss_vit.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                print('[%d, %5d] loss_vit: %.3f' %
                    (epoch + 1, i + 1, running_loss_vit / 100))
                running_loss = 0.0
                running_loss_vit = 0.0
        losses[epoch]=running_loss
        losses_v[epoch]=running_loss_vit

    time_f=time.time()
    plt.plot(losses,'-o',label="CNN")
    plt.plot(losses_v,'-d',label="ViT")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    
    print('Finished Training')
    print(f"Training time: {time_f-time_i}")
