import sys

sys.path.insert(0,'/Users/annabearden/CHE4230/Asrc')

import AdataPreprocessing

from AdataPreprocessing import AADataPreprocessing

def Atest():
    preprocessor = AADataPreprocessing()
    data = preprocessor.load_data()

    return None
