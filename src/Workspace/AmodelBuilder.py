from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn

import sys

sys.path.append('/Users/annabearden/CHE4230/Asrc')

import AdataPreprocessing
from AdataPreprocessing import AADataPreprocessing

sys.path.append('/Users/annabearden/CHE4230/Asrc/ann.py')


from ann import ANN, Net

from Aplotters import Plotters


class ModelBuilder(AADataPreprocessing, Plotters, ANN, Net):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def mlp(self, X_train, X_test, y_train, y_test):
        #Create MLP model
        MLP_classifier = MLPClassifier(hidden_layer_sizes=(200, ),
                                       max_iter=3000,
                                       verbose=False,
                                       learning_rate_init=0.0001,
                                       random_state=1)

        #Train the model
        MLP_classifier.fit(X_train, y_train)

        #Test the model
        MLP_predicted = MLP_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(MLP_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        MLP_accuracy = accuracy_score(y_test, MLP_predicted)

        return MLP_classifier

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        DT_classifier = DecisionTreeClassifier()

        #Train the model
        DT_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = DT_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(DT_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        DT_accuracy = accuracy_score(y_test, DT_predicted)

        #get performance
        DT_accuracy = accuracy_score(y_test, DT_predicted)

        return DT_classifier

    def ann(self, X_train, X_test, y_train, y_test):
        ann_classifier = ANN()
        model = Net()
        print(model)

        trainloader, testloader = ann_classifier.dataloader(
            X_train, X_test, y_train, y_test)

        train_loss = ann_classifier.train(model, trainloader)

        accuracy = ann_classifier.accuracy2(model, testloader)

        return ann_classifier, train_loss