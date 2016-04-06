from libs import store, models
from libs.features import *
from config import *
import xlsxwriter
import math
from scipy.stats import norm

def exc():
    alpha = 0.05
    tt = norm.isf(alpha / 2)

    modelInfos, modelDict = store.loadAllModel()

    groups = models.groupModels(modelInfos, modelDict)

    data = [['#', 'Learning Rate', 'Learning Rule', 'N_Iter', 'Features', 'Avg_Mean_Distance', 'Avg_Var_Distance',
             'alpha', 'H0: Avg_Mean_Distance = 0']]

    tmp = store.loadTermData()
    termList = {'X': [i[0] for i in tmp], 'y': [int(i[1]) for i in tmp]}

    for i, igroup in zip(range(len(groups)), groups):
        if (len(igroup['models'].keys()) >= 2):
            feature_manager.updateFeatureList(igroup['group-info']['features'])
            _X = np.asarray([extractFeatureText(term, getFeatureNames()) for term in termList['X']])

            d, v = models.checkModelConvergence(igroup['models'], _X)
            t = d / math.sqrt(v)

            data.append([
                i,
                igroup['group-info']['learning_rate'],
                igroup['group-info']['learning_rule'],
                igroup['group-info']['n_iter'],
                str(igroup['group-info']['features']),
                d,
                v,
                alpha,
                'Accept' if abs(t) < tt else 'Reject'
            ])

    # ================================
    workbook = xlsxwriter.Workbook(folder_model + '/' + file_model_result)
    store.writeSheet(workbook.add_worksheet('original'), data)
    workbook.close()



