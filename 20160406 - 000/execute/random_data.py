__author__ = 'Thong_Le'

from libs import store
from libs.features import extractFeature, getFeatureNames, randomSample

def exc():
    # store.loadFeatureCSV()
    # 4.1 Random Training and Testing Data
    x = randomSample()
    # store.saveTrainingTestingData(x)
    store.saveTrainingTestingDataCSV(x, getFeatureNames())