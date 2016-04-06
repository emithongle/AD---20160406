__author__ = 'Thong_Le'

import datetime
import time

import xlrd
import xlsxwriter

from config import *
from libs.features import getFeatureNames
from libs.models import modelDetails
from libs.segment import getTemplateRemoveFilters
from libs.store import writeSheet

# execution
from execute import preprocessing, extract_feature, random_data, train, \
    test_term_classifier_model as tt, test_address_segment as tas, \
    test_model_groups as tmg
# ==================

tmpsheet = xlrd.open_workbook('running_logs/logs.xlsx').sheet_by_index(0)
logs = []
for i in range(tmpsheet.nrows):
    logs.append(tmpsheet.row_values(i))

for i in range(nrun):

    t = datetime.datetime.now()
    timeManage.setTime(str(t.date().strftime('%Y%m%d')) + '_' + str(t.time().strftime('%H%M%S')))

    millis_S = int(round(time.time() * 1000))

    feature_manager.randomFeatures()

    # 1. Preprocessing
    preprocessing.exc()

    # 2. Extract Features
    extract_feature.exc()

    # 3. Random Data
    random_data.exc()

    # 4. Train Data
    tacc = train.exc()

    # 5. Test
    # 5.1. Test Term Classification
    ttacc = tt.exc()
    # 5.2. Test Address Segmentation
    acc = tas.exc()

    millis_E = int(round(time.time() * 1000))

    features = getFeatureNames()
    tfilters = getTemplateRemoveFilters()

    logs.append([timeManage.getTime(),
                 (millis_E - millis_S)/1000,
                 ', '.join([key for key in preprocessing_name if preprocessing_name[key]])
                    if bpreprocessing else '',
                 str(len(features)) + ' features: ' + ', '.join(features),
                 model_type,
                 modelDetails(),
                 tacc,
                 ttacc,
                 str(len(tfilters)) + ' filters: ' + ', '.join(tfilters),
                 acc]
            )

workbook = xlsxwriter.Workbook(folder_running_logs + '/' + file_log)
writeSheet(workbook.add_worksheet('logs'), logs)
workbook.close()

tmg.exc()