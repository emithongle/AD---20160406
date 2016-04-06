__author__ = 'Thong_Le'

import libs.segment as sg
import libs.store as store
from libs.features import getFeatureNames
from config import *

import numpy as np
import matplotlib.pyplot as plt

def exc(file='testdata.txt', label_file=folder_datasource + '/' + 'fa_labels'):
    # 5. Test & Labels
    textList = store.loadTextData(file=file)
    labels = store.loadCSV(label_file)

    if (file_model):
        _time, templateList = sg.parseAddress(textList, file_model)
    else:
        _time, templateList = sg.parseAddress(textList)

    acc, dist = checkAddressSegmentAccuracy(labels, templateList)

    titles = [['#', 'Text', 'Runtime'],
              ['TestCase', 'Top', 'Name', 'Address', 'Phone', 'NameScore', 'AddressScore', 'PhoneScore', 'Score',
               'PrepName'] + ['Name_' + ft for ft in getFeatureNames()] + \
               ['PrepAddress'] + ['Address_' + ft for ft in getFeatureNames()] + \
               ['PrepPhone'] + ['Phone_' + ft for ft in getFeatureNames()]]

    if (file_model):
        store.saveResults(titles, (textList, _time), templateList, file=file_model + '_' + file_segment_address_result,
                            acc=acc)
    else:
        store.saveResults(titles, (textList, _time), templateList, file=timeManage.getTime() + '_' + file_segment_address_result,
                            acc=acc)

    return acc

def getRankOfTemplate(ilabel, templates):
    for i, template in zip(range(len(templates)), templates):
        if (ilabel[0] == template['name']['term'] and
            ilabel[1] == template['address']['term'] and
            ilabel[2] == template['phone']['term']):
            return i
    return -1

def checkAddressSegmentAccuracy(labels, templateList):
    acc = 0
    # dist = {}
    ids = []
    for (ilabels, templates) in zip([i[1:] for i in labels[1:]], templateList):
        id = getRankOfTemplate(ilabels, templates)
        if (id >= 0):
            ids.append(id)

    if (0 in ids):
        acc = sum([1 if (i == 0) else 0 for i in ids]) / len(ids)

    if (file_model):
        plotHistogram(ids, save=True, name=folder_results + '/' + file_model)
    else:
        plotHistogram(ids, save=True, name=folder_results + '/' + timeManage.getTime())

    return acc, ids

def plotHistogram(_x, save=True, name='image'):
    x = np.asarray(_x)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # the histogram of the data
    n, bins, patches = ax.hist(x, 100)

    ax.set_xlabel('Top')
    ax.set_ylabel('Probability')
    ax.set_title(name)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, max(n))
    ax.grid(True)

    if (save):
        plt.savefig(name + '.png')
    # plt.show()