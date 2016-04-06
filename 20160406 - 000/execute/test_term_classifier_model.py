__author__ = 'Thong_Le'

import numpy as np

from config import *
from libs.features import extractFeatureText, getFeatureNames, preprocess
from libs import store


def exc():
    # 5. Test
    tmp = store.loadTermData()
    termList = {'X': [i[0] for i in tmp], 'y': [int(i[1]) for i in tmp]}

    print('=======================================================')
    print('=> Term Classifying...')

    if (file_model):
        clf = store.loadClassifier(file=file_model)
    else:
        clf = store.loadClassifier()
    results = []

    ft = getFeatureNames()

    for i in range(len(termList['X'])):
        preprocessd_term = preprocess(termList['X'][i])
        X = np.asarray([extractFeatureText(termList['X'][i], ft)])
        y_hat = clf.predict(X)[0].tolist()[0]
        results.append(clf.predict(X)[0].tolist() + clf.predict_proba(X)[0].tolist() +
                       [1 if (y_hat != termList['y'][i]) else 0, preprocessd_term] + X[0].tolist())

    titles = ['TestCase', 'Term', 'Label', 'Predicted Label', 'Name Score', 'Address Score', 'Phone Score', 'Error', 'Preprocessed_Term'] + ft

    tacc = sum([1 for (y1, y2) in zip(termList['y'], [result[0] for result in results]) if (y1 == y2)]) / len(termList['y'])

    if (file_model):
        store.saveTermTestResults(tacc, titles, termList, results, file=file_model + '_' + file_term_classify_result)

    else:
        store.saveTermTestResults(tacc, titles, termList, results, file=timeManage.getTime() + '_' + file_term_classify_result)

    return tacc