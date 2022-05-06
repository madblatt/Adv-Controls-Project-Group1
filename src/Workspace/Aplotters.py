import matplotlib.pyplot as plt
from sklearn import tree


class Plotters():
    def __init__(self):
        pass

    def plotLoss(self, model):
        plt.plot(model.loss_curve_)
        plt.show()

        return None

    def plotTree(self, model):

        tree.plot_tree(model)
        plt.show()

        return None

    def plotANN(self, loss):

        plt.plot(loss)
        plt.show()

        return None
