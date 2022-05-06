
import sys

sys.path.insert(0,'/Users/annabearden/CHE4230/Asrc')

import AdataPreprocessing

from AdataPreprocessing import AADataPreprocessing

def test():
    preprocessor = AADataPreprocessing()
    data = preprocessor.load_data()

    X_train, X_test, y_train, y_test = preprocessor.split_data(data)

    return None
