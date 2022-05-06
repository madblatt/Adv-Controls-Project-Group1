import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class AADataPreprocessing():
    def __init__(self):
        pass

    def load_data(self):

        data = pd.read_csv("Adata/RawData.csv", sep = "/t", header = None, engine='python',).dropna()
        print(data.head(10))

        data = data.to_numpy()

        return data
    def split_data(self, data):

        x = data[:, -1]
        y = data[:, -1]

        #Data preprocessing
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        #Create train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.3,
                                                            shuffle=True)

        return X_train, X_test, y_train, y_test

    def split_data_ann(self, data):
        x = data[:, :-1]
        y = data[:, -1]

        #We need classes to be 0 and 1
        for i in range(len(y)):
            if y[i] == 1:
                y[i] = 0

            else:
                y[i] = 1

        #Data preprocessing
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        #Create train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.3,
                                                            shuffle=True)

        return X_train, X_test, y_train, y_test