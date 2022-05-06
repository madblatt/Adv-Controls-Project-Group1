
import sys

sys.path.insert(0,'/Users/annabearden/CHE4230/Asrc')

import AmodelBuilder

from AmodelBuilder import ModelBuilder

def test():

    builder = ModelBuilder()

    data = builder.load_data()

    X_train, X_test, y_train, y_test = builder.split_data(data)

    model = builder.dt(X_train, X_test, y_train, y_test)

    builder.plotTree(model)

    return None