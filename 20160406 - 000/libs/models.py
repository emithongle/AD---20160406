__author__ = 'Thong_Le'

from sknn.mlp import Classifier, Layer
import numpy as np
from config import model_type, model_config

def buildClassifer(name='Neuron Network'):
    model = None
    if (name=='Neuron Network'):
        model = Classifier(
            layers=[Layer(model_config['layers'][i][1], units=model_config['layers'][i][0])
                        for i in range(len(model_config['layers']))],
            learning_rule=model_config['learning_rule'],
            learning_rate=model_config['learning_rate'],
            n_iter=model_config['n_iter']
        )
    return model

def modelDetails():
    if (model_type == 'Neuron Network'):
        return str(len(model_config['layers'])) + ' layers: [' + \
                ', '.join([str(n_unit) + '-' + act_func for (n_unit, act_func) in model_config['layers']]) + \
                '], learning_rate: ' + str(model_config['learning_rate']) + ', learning_rule: ' + model_config['learning_rule'] + \
                ', n_iterator: ' + str(model_config['n_iter'])

# def checkSimilarModel(mi1, mi2):
#     return False

def groupModels(modelInfos, models):
    groups = []

    for modelInfo in modelInfos:
        group_info = {
            'learning_rate': modelInfo['model']['config']['learning_rate'],
            'learning_rule': modelInfo['model']['config']['learning_rule'],
            'n_iter': modelInfo['model']['config']['n_iter'],
            'features': modelInfo['features']
        }

        flag = True
        for igroup in groups:
            if (igroup['group-info'] == group_info):
                flag = False
                igroup['models'][modelInfo['name']] = models[modelInfo['name']]

        if (flag):
            groups.append({
                'group-info': group_info,
                'models': { modelInfo['name'] : models[modelInfo['name']] }
            })

    return groups

def distantProb(prob_1, prob_2):
    tmp = np.power(prob_1 - prob_2, 2)
    return tmp.mean(), tmp.var()

def checkModelConvergence(models, _X):

    probX = {mkey: models[mkey].predict_proba(_X) for mkey in models}

    d, v = 0, 0
    for mkey_1 in probX:
        for mkey_2 in probX:
            if (mkey_1 != mkey_2):
                td, tv = distantProb(probX[mkey_1], probX[mkey_2])
                d += td
                v += tv

    nmodels = len(probX.keys())
    if (nmodels == 2):
        return d, v
    return d / ((nmodels - 1) * (nmodels - 2)), v / ((nmodels - 1) * (nmodels - 2))