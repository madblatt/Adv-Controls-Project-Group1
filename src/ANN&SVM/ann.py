#Import packages
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


class ANN():
    def __init__(self):
        pass

    #Dataloader
    def dataloader(self, X_train, X_test, y_train, y_test):
        batch_size = 10

        trainset = []
        for i in range(len(X_train)):
            trainset.append([X_train[i], y_train[i]])

        testset = []
        for i in range(len(X_test)):
            testset.append([X_test[i], y_test[i]])

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        return trainloader, testloader

    def train(self, net, trainloader):
        train_loss = []

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        epochs = 50

        for epoch in range(epochs):
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):

                x, labels = data

                optimizer.zero_grad()

                outputs = net(x.float())

                loss = criterion(outputs, labels.long())

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            loss = running_loss / len(trainloader)
            train_loss.append(loss)

            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch + 1, epochs, loss))

        return train_loss

    def accuracy2(self, model, testloader):
        model.eval(
        )  #We are not training anymore so we now use .eval() to make predictions without altering the weights

        with torch.no_grad():
            correct = 0
            total = 0
            for x, labels in testloader:
                outputs = model(x.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.float()).sum().item()

            print('Test Accuracy of the model: {} %'.format(
                (correct / total) * 100))

        return (correct / total) * 100


#Build a model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Traditional NNs (fully connected layers)
        self.fc0 = nn.Linear(in_features=36, out_features=200)
        self.fc1 = nn.Linear(in_features=200, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
