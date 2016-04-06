__author__ = 'Thong_Le'

from config import *
from libs.store import loadJson, saveJson
from libs.features import getFeatureNames

def getModelConfig(testAcc=0, valAcc=0):
    dct = {}

    dct['name'] = timeManage.getTime()

    dct['dictionary'] = {
        'folder': folder_datasource,
        'file': [files_dictionary]
    }

    dct['database'] = {
        'folder': folder_data,
        'name': files_data[0],
        'address': files_data[1],
        'phone': files_data[2]
    }

    dct['preprocessing'] = {
        'flag': bpreprocessing,
        'type': list(preprocessing_name.keys())
    }

    dct['features'] = getFeatureNames()

    dct['model'] = {
        'class': model_type,
        'target': model_target,
        'config': model_config
    }

    dct['results'] = {
        'test-accuracy': testAcc,
        'validate-accuracy': valAcc
    }

    return dct

def updateModelConfig(folder=folder_model, file=file_model_info, testAcc=0, valAcc=0):
    file_path = folder + '/' + file

    modelConfig = loadJson(file_path)
    modelConfig.append(getModelConfig(testAcc, valAcc))
    saveJson(modelConfig, file_path)