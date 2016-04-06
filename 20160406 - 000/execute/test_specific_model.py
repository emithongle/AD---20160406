__author__ = 'Thong_Le'

import numpy as np

from libs import store
from libs.features import extractFeature, randomSample
from libs.models import buildClassifer
from config import *


# 1. Read txt data
tupleData = store.loadTxtData()

# 2. Extract features
featureTuples = extractFeature(feature_func)
store.saveFeature(featureTuples)

accs = []

if (standard_data):
    X_train, y_train, X_test, y_test = randomSample()
    X_train, y_train, X_test, y_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)

    print('=> Testing model with static data...')
    for i in range(nTesting):
        # 3.3 Train model
        print('==> Training model ' + str(i) + '...')
        clf_model = buildClassifer()
        clf_model.fit(X_train, y_train)

        y_hat = clf_model.predict(X_test)
        tacc = sum([1 for (y1, y2) in zip(y_test, y_hat) if (y1 == y2)]) / len(y_hat)
        print('<== Acc = ', tacc)
        accs.append(tacc)
else:
    print('=> Testing model with random data...')
    for i in range(nTesting):
        X_train, y_train, X_test, y_test = randomSample()
        X_train, y_train, X_test, y_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)

        # 3.3 Train model
        print('==> Training model ' + str(i) + '...')
        clf_model = buildClassifer()
        clf_model.fit(X_train, y_train)

        y_hat = clf_model.predict(X_test)
        tacc = sum([1 for (y1, y2) in zip(y_test, y_hat) if (y1 == y2)]) / len(y_hat)
        print('<= Acc = ', tacc)
        accs.append(tacc)

store.saveTestModel(accs)