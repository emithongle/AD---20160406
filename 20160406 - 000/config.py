__author__ = 'Thong_Le'

import json
import datetime
import itertools as its
import random as rd

def readFile(file):
    strList = []
    infile = open(file, encoding="utf-8")
    for line in infile:
        strList.append(line)
    return strList

def loadJson(file):
    try:
        return json.loads(''.join(readFile(file + '.json')))
    except:
        return None

class TimeManage():
    def __init__(self):
        self.time = ''

    def setTime(self, _time):
        self.time = _time

    def getTime(self):
        return self.time

timeManage = TimeManage()

# ================================

folder_datasource = '0. Data-source'
files_dictionary = 'dictionary' # json

folder_data = '1. Data'
files_data = ['name.txt', 'address.txt', 'phone.txt']

folder_preprocessing = '2. Preprocessing'
files_preprocessed = ['preprocessed_name', 'preprocessed_address', 'preprocessed_phone'] # csv

folder_features = '3. Features'
files_features = ['name_features', 'address_features', 'phone_features'] # csv
files_traintest = ['training_data', 'testing_data']

folder_model = '4. Model'
file_model = None # '20160329_150346'

folder_test = '5. Test'
file_full_address_test = 'testdata.txt'
file_term_test = 'termdata' # csv

folder_results = '6. Results'
file_segment_address_result = 'full_address_results.xlsx'
file_term_classify_result = 'term_results.xlsx'
file_model_result = 'test_model_results.xlsx'

folder_running_logs = 'running_logs'
file_log = 'logs.xlsx' # csv

# ==============================================

# number of executive run.py
nrun = 30

tmp = loadJson(folder_datasource + '/' + files_dictionary)
nameTermSet = tmp['name-term-set']
addressTermSet = tmp['address-term-set']
phoneTermSet = tmp['phone-term-set']

asi = tmp['ascii']
unic = tmp['unicode']
upchars = tmp['upper-characters']

feature_func = 'feature'
preprocessing_func = 'preprocessing'


# Preprocessing
bpreprocessing = True
preprocessing_name = {'convert unicode to ascii': True, 'convert to lower': True,
                      'remove break line': False, 'remove multiple spaces': True,
                      'trim "space" and ","': True}

# Features
feature_list_all = [
    ('length', True),
    # ('#ascii', True),
    # ('#digit', True),
    # ('#punctuation', True),

    ('#ascii/(#ascii+#digit+#punctuation)', True),
    ('#digit/(#ascii+#digit+#punctuation)', True),

    # ('%ascii', True),
    # ('%digits', True),

    # ('%kwName', True),
    # ('%kwAddress', True),
    # ('%kwPhone', True),
    (['%kwName', '%kwAddress', '%kwPhone'], True),


    ('%max_digit_skip_0', False),
    ('#max_digit_skip_0', False),
    ('%max_digit_skip_0_1', False),
    ('#max_digit_skip_0_1', False),

    ('#max_digit_skip_0_2', False),  # max_digit (skip = {" ", "."}) : 0 : 0, 1: 1 - 7, 2: >= 8
    (['#max_digit_skip_0_2_0',
      '#max_digit_skip_0_2_1',
      '#max_digit_skip_0_2_2'], True),

    ('first_character_ascii', False),
    ('first_character_digit', False),
    ('first_character_type', False),     # first_char: 1: ascii, 2: digit, 3: {"(", "+"}, 0: others
    (['first_character_type_0',
      'first_character_type_1',
      'first_character_type_2',
      'first_character_type_3'], True),

    ('#(', False),
    ('#+', False),
    ('#/', False),

    ('#"space"', True)
]

class FeatureManager():
    def __init__(self, fts):
        self.features = [ft for ft in fts if (ft[1])]
        self.nfeatures = len(self.features)

        self.combinations = []
        for i in range(self.nfeatures):
            self.combinations += list(its.combinations(self.features, i + 1))

        self.curFeatures = []

    def randomFeatures(self):
        self.curFeatures = list(self.combinations[rd.randint(0, len(self.combinations) - 1)])

    def getFeatureList(self):
        return self.curFeatures

    def updateFeatureList(self, fts):
        self.curFeatures = [(ft, True) for ft in fts]



feature_manager = FeatureManager(feature_list_all)

# Model
model_type = 'Neuron Network'
model_target = 'Classify Name/Address/Phone'

model_config = {
    'layers': [(100, 'Sigmoid'), (3, 'Softmax')],
    'learning_rate': 0.01,
    'learning_rule': 'adagrad',
    'n_iter': 10
}
file_model_info = 'model_info'


# Templates
template_rm_filters = {
    'Phone: #digit < 8': False,
    'Phone: 2 * _%ascii < _%digit': False,
    'Phone: _%ascii > 0 & %kwPhone = 0': False,
    'Phone: first_character_type != digit': False,

    'Name: #ascii < 5': False,
    'Name: _%digit > 0': False,
    'Name: first_character_type != ascii': False,
    'Name: _%ascii + _%digit < 1': False
}


# Testing Model
standard_data = True
nTesting = 10

skip_punctuation = ' .'
rm_preprocessed_punctuation = """ ,"""
