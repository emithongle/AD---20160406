__author__ = 'Thong_Le'

import string

from sklearn.cross_validation import train_test_split
import numpy as np

from config import *
from libs import store
import string
import re

def extractFeature(tupledata):
    names, addresses, phones = tupledata

    y_name, X_name = names[0], []
    y_address, X_address = addresses[0], []
    y_phone, X_phone = phones[0], []

    ft = getFeatureNames()

    print('=======================================================')
    print('=> Feature extracting...')

    print('==> Name feature extracting...')
    for text in names[1]:
        X_name.append(feature(text[0], ft))

    print('<== Name feature extracted.')

    print('==> Address feature extracting...')
    for text in addresses[1]:
        X_address.append(feature(text[0], ft))
    print('<== Address feature extracted.')

    print('==> Phone feature extracting...')
    for text in phones[1]:
        X_phone.append(feature(str(text[0]), ft))
    print('<== Phone feature extracted.')

    # return [{'X': X_name, 'y': y_name}, {'X': X_address, 'y': y_address}, {'X': X_phone, 'y': y_phone}]
    return [[y_name, X_name, names[1]], [y_address, X_address, addresses[1]], [y_phone, X_phone, phones[1]]]


def extractFeatureText(text, features):
    if (bpreprocessing):
        text = preprocess(text)
    return feature(text, features)

def randomSample(tupleData=None, testSize=0.2):
    print('=======================================================')
    print('=> Randoming data...')

    if (tupleData):
        X_train_names, X_test_names, y_train_name, y_test_name = \
            train_test_split(np.asarray(tupleData[0]['X']), tupleData[0]['y'], test_size=testSize)

        X_train_address, X_test_address, y_train_address, y_test_address = \
            train_test_split(np.asarray(tupleData[1]['X']), tupleData[1]['y'], test_size=testSize)

        X_train_phone, X_test_phone, y_train_phone, y_test_phone = \
            train_test_split(np.asarray(tupleData[2]['X']), tupleData[2]['y'], test_size=testSize)

    else:

        datatuple = store.loadFeatureCSV()

        X_train_names, X_test_names, y_train_name, y_test_name = \
            train_test_split(np.asarray(datatuple[0][1]), datatuple[0][0], test_size=testSize)

        X_train_address, X_test_address, y_train_address, y_test_address = \
            train_test_split(np.asarray(datatuple[1][1]), datatuple[1][0], test_size=testSize)

        X_train_phone, X_test_phone, y_train_phone, y_test_phone = \
            train_test_split(np.asarray(datatuple[2][1]), datatuple[2][0], test_size=testSize)

    X_train = np.append(np.append(X_train_names.tolist(),X_train_address.tolist(), axis=0), X_train_phone, axis=0)
    y_train = y_train_name + y_train_address + y_train_phone

    X_test = np.append(np.append(X_test_names.tolist(), X_test_address.tolist(), axis=0), X_test_phone, axis=0)
    y_test = y_test_name + y_test_address + y_test_phone

    print('=> Randomed data.')

    return (X_train, y_train, X_test, y_test)

def removeDuplicate(termList):
    t = []
    for i in range(len(termList)):
        b = True
        for j in range(len(t)):
            if ((' ' + t[j] + ' ').find(' ' + termList[i] + ' ') >= 0):
                b = False
                break
            elif ((' ' + termList[i] + ' ').find(' ' + t[j] + ' ') >= 0):
                t[j] = termList[i]
                b = False
                break
        if (b):
            t.append(termList[i])

    return t

def preprocess4GetTerm(text):
    for c in string.digits:
        text = text.replace(c, ' ')

    for c in string.punctuation:
        text = text.replace(c, ' ')

    return text

def checkCharacterType(c):
    if c in string.digits:
        return {True: 'digit'}
    elif c in string.ascii_uppercase + string.ascii_lowercase + unic:
        return {True: 'ascii'}
    else:
        return {True: None}

def findMaxStringP(text, type, skip=0, skip_chars=''):
    i, nskip = 0, 0
    while (i < len(text) and nskip <= skip):
        if (type == 'ascii'):
            if (text[i] not in skip_chars):
                if (text[i] in string.digits + string.punctuation):
                    nskip += 1
                elif (text[i] in string.ascii_letters):
                    nskip = 0
        elif (type == 'digit'):
            if (text[i] not in skip_chars):
                if (text[i] in string.ascii_letters + string.punctuation):
                    nskip += 1
                elif (text[i] in string.digits):
                    nskip = 0
        elif (type == 'punctuation'):
            None
        i += 1
    if nskip > skip:
        return text[:i-1].strip()
    elif (i == len(text)):
        return text.strip()

    return text[:i].strip()

def findMaxString(text, skip=0, skip_chars=''):
    tc, td = '', ''
    # c = 0
    try:
        for i in range(len(text)):
            if text[i] in string.digits:
                t = findMaxStringP(text[i:], 'digit', skip, skip_chars)
                # if (c > 0):
                #     t = text[i-c:i] + t
                #     c = 0
                if (len(td) < len(t)):
                    td = t

            elif text[i] in string.ascii_letters:
                t = findMaxStringP(text[i:], 'ascii', skip, skip_chars)
                # if (c > 0):
                #     t = text[i-c:i] + t
                #     c = 0
                if (len(tc) < len(t)):
                    tc = t

            # elif (text[i] in skip_punctuation): # text[i] in string.punctuation and
            #     # t = findMaxStringP(text[i:], 'punctuation', skip, split_chars)
            #     c += 1
            # else:
            #     c = 0

    except ValueError:
        None

    return tc, td

def getFeatureNames(fl = None):
    fl = feature_manager.getFeatureList()
    l = []

    for i in range(len(fl)):
        if (fl[i][1]):
            if (type(fl[i][0]).__name__ == 'list'):
                l += fl[i][0]
            else:
                l.append(fl[i][0])

    return l

def feature(text, fl):
    textLen = text.__len__()

    # preprocessText = preprocess4GetTerm(text)
    # preprocessedTermList = preprocessText.split()
    # preprocessedTermListLen = len(preprocessedTermList)

    nameTerms = removeDuplicate([term for term in nameTermSet if (text.find(term)>= 0)])
    addressTerms = removeDuplicate([term for term in addressTermSet if (text.find(term)>= 0)])
    phoneTerms = removeDuplicate([term for term in phoneTermSet if (text.find(term)>= 0)])

    nascii = sum([1 for c in text if (c in string.ascii_letters)])
    ndigit = sum([1 for c in text if (c in string.digits)])
    npunctuation = sum([1 for c in text if (c in string.punctuation)])

    ft = []

    for i in range(len(fl)):
        if (fl[i] == 'length'):
            ft += [len(text)]

        elif (fl[i] == '#ascii'):
            ft += [nascii]
        elif (fl[i] == '#digit'):
            ft += [ndigit]
        elif (fl[i] == '#punctuation'):
            ft += [npunctuation]

        elif (fl[i] == '#ascii/(#ascii+#digit+#punctuation)'):
            ft += [nascii/(nascii + ndigit + npunctuation)]

        elif (fl[i] == '#digit/(#ascii+#digit+#punctuation)'):
            ft += [ndigit/(nascii + ndigit + npunctuation)]

        elif (fl[i] == '%ascii'):
            ft += [sum([1 for c in text if (c in string.ascii_letters)]) / textLen]

        elif (fl[i] == '%digits'):
            ft += [sum([1 for c in text if (c in string.digits)]) / textLen]


        elif (fl[i] == '%kwName'):
            ft += [len(nameTerms) / len(nameTermSet) if (len(nameTermSet) > 0) else 0]
        elif (fl[i] == '%kwAddress'):
            ft += [len(addressTerms) / len(addressTermSet) if (len(addressTermSet)) else 0]
        elif (fl[i] == '%kwPhone'):
            ft += [len(phoneTerms) / len(phoneTermSet) if (len(phoneTermSet)) else 0]

        elif (fl[i] == '%max_digit_skip_0'):
            tc_0, td_0 = findMaxString(text, 0)
            ft += [len(td_0) / textLen]
        elif (fl[i] == '#max_digit_skip_0'):
            tc_0, td_0 = findMaxString(text, 0)
            ft += [len(td_0)]
        elif (fl[i] == '%max_digit_skip_0_1'):
            tc_0, td_0 = findMaxString(text, 0, skip_punctuation)
            ft += [len(td_0) / textLen]
        elif (fl[i] == '#max_digit_skip_0_1'):
            tc_0, td_0 = findMaxString(text, 0, skip_punctuation)
            ft += [len(td_0)]

        elif (fl[i] == '#max_digit_skip_0_2'):
            tc_0, td_0 = findMaxString(text, 0, skip_punctuation)
            ft += [0 if (len(td_0) == 0) else
                  1 if (1 <= len(td_0) and (len(td_0) <= 7)) else
                  2]

        elif (fl[i] == '#max_digit_skip_0_2_0'):
            tc_0, td_0 = findMaxString(text, 0, skip_punctuation)
            ft += [1 if (len(td_0) == 0) else 0]
        elif (fl[i] == '#max_digit_skip_0_2_1'):
            tc_0, td_0 = findMaxString(text, 0, skip_punctuation)
            ft += [1 if (1 <= len(td_0) and len(td_0) <= 7) else 0]
        elif (fl[i] == '#max_digit_skip_0_2_2'):
            tc_0, td_0 = findMaxString(text, 0, skip_punctuation)
            ft += [1 if (8 <= len(td_0)) else 0]

        elif (fl[i] == 'first_character_ascii'):
            if (len(text)>= 1):
                ft += [1 if checkCharacterType(text[0])[True] == 'ascii' else 0]
            else:
                ft += [0]

        elif (fl[i] == 'first_character_digit'):
            if (len(text)>= 1):
                ft += [1 if checkCharacterType(text[0])[True] == 'digit' else 0]
            else:
                ft += [0]

        elif (fl[i] == 'first_character_type'):
            ft += [
                1 if (text[0] in string.ascii_letters) else \
                2 if (text[0] in string.digits) else \
                3 if (text[0] in '(+') else \
                0
            ]

        elif (fl[i] == 'first_character_type_0'):
            ft += [0 if (text[0] in string.ascii_letters + string.digits + '(+') else 1]
        elif (fl[i] == 'first_character_type_1'):
            ft += [1 if (text[0] in string.ascii_letters) else 0]
        elif (fl[i] == 'first_character_type_2'):
            ft += [1 if (text[0] in string.digits) else 0]
        elif (fl[i] == 'first_character_type_3'):
            ft += [1 if (text[0] in '(+') else 0]

        elif (fl[i] == 'last_character_ascii'):
            if (len(text)>= 1):
                ft += [1 if checkCharacterType(text[-1])[True] == 'ascii' else 0]
            else:
                ft += [0]

        elif (fl[i] == 'last_character_digit'):
            if (len(text)>= 1):
                ft += [1 if checkCharacterType(text[-1])[True] == 'digit' else 0]
            else:
                ft += [0]
        elif (fl[i] == '#('):
            ft += [sum([1 for c in text if '(' == c])]
        elif (fl[i] == '#+'):
            ft += [sum([1 for c in text if '+' == c])]
        elif (fl[i] == '#/'):
            ft += [sum([1 for c in text if '/' == c])]

        elif (fl[i] == '#"space"'):
            ft += [sum([1 for c in text if ' ' == c])]
    return ft

def dataPreprocess(tupleData):

    for i in range(len(tupleData[0][1])):
        tupleData[0][1][i] = preprocess(tupleData[0][1][i])

    for i in range(len(tupleData[0][1])):
        tupleData[1][1][i] = preprocess(tupleData[1][1][i])

    for i in range(len(tupleData[0][1])):
        tupleData[2][1][i] = preprocess(tupleData[2][1][i])

    return tupleData

def preprocess(text):
    # print(text.encode())
    if (preprocessing_name['convert unicode to ascii']):
        for i in range(len(unic)):
            text = text.replace(unic[i], asi[i])

    if (preprocessing_name['remove break line']):
        text = text.replace('\n', '')

    if (preprocessing_name['convert to lower']):
        text = text.lower()

    if (preprocessing_name['remove multiple spaces']):
        text = re.sub(' +', ' ', text)

    if (preprocessing_name['trim "space" and ","']):
        text = text.strip(rm_preprocessed_punctuation)

    return text