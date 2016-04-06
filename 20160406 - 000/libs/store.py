__author__ = 'Thong_Le'

import pickle
import csv
from os import listdir, curdir
from os.path import isfile, join

import numpy as np
import xlsxwriter

from config import *


def readFile(file):
    strList = []
    infile = open(file, encoding="utf-8")
    for line in infile:
        strList.append(line)
    return strList

def saveJson(data, file):
    with open(file+'.json', 'w', encoding='utf8') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)

def loadJson(file):
    try:
        return json.loads(''.join(readFile(file + '.json')))
    except:
        return None

def saveCSV(data, file):
    with open(file + '.csv', 'w', newline='', encoding="utf-8") as f:
    # with open(file + '.csv', 'w', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)
        # for row in data:
        #     writer.writerow(row)

def loadCSV(file):
    data = []
    with open(file + '.csv', newline='', encoding="utf-8") as f:
    # with open(file + '.csv', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)

    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         if (data[i][j].find('.') > 0):
    #             try:
    #                 data[i][j] = float(data[i][j])
    #             except ValueError:
    #                 None
    #         else:
    #             try:
    #                 data[i][j] = int(data[i][j])
    #             except ValueError:
    #                 None

    return data

def loadTxtData(folder=folder_data, files=files_data):

    print('=======================================================')
    print('=> Reading data...')

    print('==> Reading name data...')
    name = readFile(folder + '/' + files[0])
    name = name[1:]
    print('<== Readed name data.')

    print('==> Reading address data...')
    address = readFile(folder + '/' + files[1])
    address = address[1:]
    print('<== Readed address data.')

    print('==> Reading phone data...')
    phone = readFile(folder + '/' + files[2])
    phone = phone[1:]
    print('<== Readed phone data.')

    return (([0] * len(name), name), ([1] * len(address), address), ([2] * len(phone), phone))

def loadTextData(folder=folder_test, file=file_full_address_test):

    print('=======================================================')
    print('=> Loading data test...')
    textList = readFile(join(curdir, folder) + '/' + file)
    textList = textList[1:]
    textList = [text[:-1] if (text[-1] == '\n') else text for text in textList]
    print('<= Loading data test...')

    return textList

def savePreprocessedDataCSV(tupledata, feature_names, folder=folder_preprocessing,
                             files=files_preprocessed):
    print('=======================================================')
    print('=> Saving preprocessing data...')

    print('==> Saving preprocessing name...')
    data = [[tupledata[0][0][i]] + [tupledata[0][1][i]] for i in range(len(tupledata[0][0]))]
    saveCSV([['Label', 'Text']] + data, folder + '/' + files[0])
    print('<== Saved preprocessing name.')

    print('==> Saving preprocessing address...')
    data = [[tupledata[1][0][i]] + [tupledata[1][1][i]] for i in range(len(tupledata[1][0]))]
    saveCSV([['Label', 'Text']] + data, folder + '/' + files[1])
    print('<== Saved preprocessing address.')

    print('==> Saving preprocessing phone...')
    data = [[tupledata[2][0][i]] + [tupledata[2][1][i]] for i in range(len(tupledata[2][0]))]
    saveCSV([['Label', 'Text']] + data, folder + '/' + files[2])
    print('<== Saved preprocessing phone.')

def loadPreprocessedDataCSV(folder=folder_preprocessing, files=files_preprocessed):
    print('=======================================================')
    print('=> Loading preprocessing data...')

    data = loadCSV(folder + '/' + files[0])
    X_name = [data[i][1:] for i in range(1, len(data))]
    y_name = [data[i][0] for i in range(1, len(data))]

    data = loadCSV(folder + '/' + files[1])
    X_address = [data[i][1:] for i in range(1, len(data))]
    y_address = [data[i][0] for i in range(1, len(data))]

    data = loadCSV(folder + '/' + files[2])
    X_phone = [data[i][1:] for i in range(1, len(data))]
    y_phone = [data[i][0] for i in range(1, len(data))]

    print('<= Loaded data.')

    return [[y_name, X_name], [y_address, X_address], [y_phone, X_phone]]


# def saveFeature(tupledata, folder=folder_features, files=files_features):
#
#     print('=======================================================')
#     print('=> Saving features...')
#
#     print('==> Saving name features...')
#     saveJson(tupledata[0], folder + '/' + files[0])
#     print('<== Saved name features.')
#
#     print('==> Saving address features...')
#     saveJson(tupledata[1], folder + '/' + files[1])
#     print('<== Saved address features.')
#
#     print('==> Saving phone features...')
#     saveJson(tupledata[2], folder + '/' + files[2])
#     print('<== Saved phone features.')

def saveFeatureCSV(tupledata, feature_names, folder=folder_features, files=files_features):

    print('=======================================================')
    print('=> Saving features...')

    print('==> Saving name features...')
    data = [[tupledata[0][0][i]] + tupledata[0][2][i] + tupledata[0][1][i] for i in range(len(tupledata[0][0]))]
    saveCSV([['Label', 'Text'] + feature_names] + data, folder + '/' + files[0])
    print('<== Saved name features.')

    print('==> Saving address features...')
    data = [[tupledata[1][0][i]] + tupledata[1][2][i] + tupledata[1][1][i] for i in range(len(tupledata[1][0]))]
    saveCSV([['Label', 'Text'] + feature_names] + data, folder + '/' + files[1])
    print('<== Saved address features.')

    print('==> Saving phone features...')
    data = [[tupledata[2][0][i]] + tupledata[2][2][i] + tupledata[2][1][i] for i in range(len(tupledata[2][0]))]
    saveCSV([['Label', 'Text'] + feature_names] + data, folder + '/' + files[2])
    print('<== Saved phone features.')

def loadFeatureCSV(folder=folder_features, files=files_features):
    print('=======================================================')
    print('=> Loading feature data...')

    data = loadCSV(folder + '/' + files[0])
    X_name = [data[i][1:] for i in range(1, len(data))]
    y_name = [data[i][0] for i in range(1, len(data))]

    data = loadCSV(folder + '/' + files[1])
    X_address = [data[i][1:] for i in range(1, len(data))]
    y_address = [data[i][0] for i in range(1, len(data))]

    data = loadCSV(folder + '/' + files[2])
    X_phone = [data[i][1:] for i in range(1, len(data))]
    y_phone = [data[i][0] for i in range(1, len(data))]

    print('<= Loaded data.')

    return [[y_name, X_name], [y_address, X_address], [y_phone, X_phone]]


# def saveTrainingTestingData(tupleData, folder='3. Features/', files=['X_train', 'y_train', 'X_test', 'y_test']):
#     print('=======================================================')
#     print('=> Saving data for training and testing...')
#
#     saveJson(tupleData[0].tolist(), folder + files[0])
#     saveJson(tupleData[1], folder + files[1])
#     saveJson(tupleData[2].tolist(), folder + files[2])
#     saveJson(tupleData[3], folder + files[3])
#
#     print('<= Saved data.')

def saveTrainingTestingDataCSV(tupleData, feature_names, folder=folder_features, files=files_traintest):
    print('=======================================================')
    print('=> Saving data for training and testing...')

    data = [[y] + x for x, y in zip(tupleData[0].tolist(), tupleData[1])]
    saveCSV([['Label', 'Text'] + feature_names] + data, folder + '/' + files[0])

    data = [[y] + x for x, y in zip(tupleData[2].tolist(), tupleData[3])]
    saveCSV([['Label', 'Text'] + feature_names] + data, folder + '/' + files[1])

    print('<= Saved data.')

# def loadTrainingTestingData(folder=folder_features, files=files_features):
#     print('=======================================================')
#     print('=> Loading data for training and testing...')
#
#     X_train = np.asarray(loadJson(folder + '/' + files[0]))
#     y_train = np.asarray(loadJson(folder + '/' + files[1]))
#     X_test = np.asarray(loadJson(folder + files[2]))
#     y_test = np.asarray(loadJson(folder + files[3]))
#
#     print('<= Loaded data.')
#
#     return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

def arrStringToArrFloat(arr):
    rarr = []
    for i in range(len(arr)):
        tmp = []
        for j in range(len(arr[i])):
            tmp.append(float(arr[i][j]))
        rarr.append(tmp)
    return rarr

def loadTrainingTestingDataCSV(folder=folder_features, files=files_traintest):
    print('=======================================================')
    print('=> Loading data for training and testing...')

    data = loadCSV(folder + '/' + files[0])
    X_train = np.asarray(arrStringToArrFloat([data[i][2:] for i in range(1, len(data))]))
    y_train = np.asarray([int(data[i][0]) for i in range(1, len(data))])

    data = loadCSV(folder + '/' + files[1])
    X_test = np.asarray(arrStringToArrFloat([data[i][2:] for i in range(1, len(data))]))
    y_test = np.asarray([int(data[i][0]) for i in range(1, len(data))])

    print('<= Loaded data.')

    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

def loadClassifier(folder=folder_model, file=None):
    print('=======================================================')
    print('=> Loading model...')

    if (file):
        file = file
    else:
        file = timeManage.getTime()

    with open(folder + '/' + file + '.pkl', 'rb') as f:
        try:
            print('=> Loading model...')
            model = pickle.load(f, encoding='latin1')
            print('=> Loaded model.')
        except:
            model = pickle.load(f)
    print('<= Loaded model.')
    return model

def saveClassifier(clf, folder=folder_model, file=None):
    print('=======================================================')
    print('=> Saving model...')
    if (file):
        file = file
    else:
        file = timeManage.getTime()

    with open(folder + '/' + file + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print('<= Saved model.')

def loadDictionary(file=folder_datasource + '/' + files_dictionary):
    tmp = loadJson(file)
    return tmp


def writeSheet(sheet, data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            sheet.write(i, j, data[i][j])


def saveResults(titles, input_tuple, templateList,
                folder=folder_results, file=file_segment_address_result, acc=tmp):

    orgData = [titles[0]]
    for i in range(len(input_tuple[0])):
        orgData.append([i, input_tuple[0][i], input_tuple[1][i]])

    sg_1 = [['Accuracy = ', acc], titles[1]]
    for i, templates in zip(range(len(templateList)), templateList):
        for j, template in zip(range(len(templates)), templates):
            t = [i, j, template['name']['term'], template['address']['term'], template['phone']['term'],
                 template['name']['score'], template['address']['score'], template['phone']['score'],
                 template['score'],
                 template['name']['preprocessed']] + template['name']['features'] + \
                 [template['address']['preprocessed']] + template['address']['features'] + \
                 [template['phone']['preprocessed']] + template['phone']['features']
            sg_1.append(t)

    # =========================

    workbook = xlsxwriter.Workbook(folder + '/' + file)
    writeSheet(workbook.add_worksheet('original'), orgData)
    writeSheet(workbook.add_worksheet('segment_1'), sg_1)
    workbook.close()

def saveTestModel(accs, folder=folder_results, file=file_model_result):

    data = [['#', 'Accuracy']]
    for i, acc in zip(range(len(accs)), accs):
        data.append([i, acc])

    workbook = xlsxwriter.Workbook(folder + '/' + file)
    writeSheet(workbook.add_worksheet('accuracy'), data)
    workbook.close()

def loadTermData(folder=folder_test, file=file_term_test):
    print('=======================================================')
    print('=> Loading term data test...')
    textList = loadCSV(folder + '/' + file)
    # textList = textList[1:]
    print('<= Loading term data test...')

    return textList

def saveTermTestResults(tacc, titles, termList, results, folder=folder_results, file=file_term_classify_result):

    data = [['Accurracy = ', tacc], titles]

    for (i, X, y, result) in zip(range(len(results)), termList['X'], termList['y'], results):
        data.append([i, X, y] + result)
    # =========================

    workbook = xlsxwriter.Workbook(folder + '/' + file)
    writeSheet(workbook.add_worksheet('result'), data)
    workbook.close()


def listFileInFolder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]

def loadAllModel(folder=folder_model):
    modelFiles = listFileInFolder(folder_model)
    models = {}
    modelInfos = loadJson(join(folder_model, file_model_info))
    for modelFile in modelFiles:
        if ('.pkl' in modelFile):
            file = modelFile[:-4]
            models[file] = loadClassifier(folder_model, file)

    return modelInfos, models