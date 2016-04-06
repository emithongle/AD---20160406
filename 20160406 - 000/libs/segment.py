__author__ = 'Thong_Le'

import time
import copy
from math import log
import numpy as np
import string

from libs import store
import libs.features as fe
from config import bpreprocessing, template_rm_filters

nterm = 3

def parseAddress(textList, file_model=None):
    if (file_model):
        clf_model = store.loadClassifier(file=file_model)
    else:
        clf_model = store.loadClassifier()
    _time = []
    templateList = []

    print('=======================================================')
    print('=> Address Parsing...')
    for (i, text) in zip(range(len(textList)), textList):
        print('==> Parsing text ' + str(i) + '...')
        millis_S = int(round(time.time() * 1000))
        ptemplates = templateSegment(text, nterm)
        templates = templateFiler(clf_model, ptemplates)
        millis_E = int(round(time.time() * 1000))
        _time.append((millis_E - millis_S)/1000)

        templateList.append(templates)

    return (_time, templateList)

def getTerm(text, l, npos, i):
    if (l != [] and npos != [] and i >= 0 and i < len(l)):
        if (i == 0):
            return text[npos[0]:npos[l[0]]]
        else:
            return text[npos[l[i-1]]:npos[l[i]]]
    return None

def buildTermList(text, l, npos):
    if (len(l) >= 1):
        lt = []
        for i in range(len(l)):
            lt.append(getTerm(text, l, npos, i))
        return lt
    return []

def calcum(l):
    if (len(l) > 0):
        cl = [l[0]]
        for i in range(len(l) - 1):
            t = cl[-1] + l[i + 1]
            cl.append(t)
        return cl
    return []

def getNext(nl, n, t):
    for i in range(n-1):
        nl[-(i + 2)] += 1
        if (sum(nl[:-(i+1)]) <= t - i - 1):
            for j in range(i+1):
                nl[-(j + 1)] = 1
            nl[-1] = t - sum(nl[:-1])
            return nl

    return None

def templateSegment(text, n):
    # split by spaces
    terms = text.replace(',', ' ').split()

    npos = []
    for i in range(len(terms)):
        if (npos == []):
            pos = text.find(terms[i])
            if (pos >= 0):
                npos.append(pos)
        else:
            pos = text.find(terms[i], npos[-1] + len(terms[i-1]))
            if (pos >= 0):
                npos.append(pos)
    npos.append(len(text)+1)

    nterms = len(terms)
    nl = [1] * (n - 1)
    nl.append(nterms - len(nl))

    segL = [buildTermList(text, calcum(nl), npos)]

    while (1):
        nl = getNext(nl, n, nterms)
        if (nl == None):
            break
        segL.append(buildTermList(text, calcum(nl), npos))

    return segL

def templateFiler(clf, ptemplates):
    templates = []

    m = {0: 'name', 1: 'address', 2: 'phone'}
    ft = fe.getFeatureNames()
    dft = {key: value for (key, value) in zip(ft, range(len(ft)))}

    for terms in ptemplates:
        # preprocessedTerms = [fe.extractFeatureText(term) for term in terms]

        preprocessedTerms = []
        for term in terms:
            if (bpreprocessing):
                preprocessedTerms.append(fe.preprocess(term))

        btmp = False
        _X = []
        for term in preprocessedTerms:
            if (len(term) > 0):
                _X.append(fe.feature(term, ft))
            else:
                btmp = True
        if (btmp):
            continue

        X = np.asarray(_X)
        cls = clf.predict(X)

        tmp = copy.deepcopy(cls.reshape((1, cls.shape[0])).tolist()[0])
        tmp.sort()
        if (tmp == list(range(len(terms)))):
            dct = {}
            probs = clf.predict_proba(X)
            for (term, cl, prob, prepTerm, _x) in zip(terms, cls, probs, preprocessedTerms, _X):
                try:
                    dct[m[int(cl)]] = {'term': term, 'score': prob[int(cl)], 'preprocessed': prepTerm, 'features': _x}
                except ValueError:
                    dct[m[int(cl)]] = {'term': term, 'score': prob[cl], 'preprocessed': prepTerm, 'features': _x}
            dct['score'] = sum([log(dct[key]['score']) for key in dct])
            if (checkTemplate(dct, dft)):
                templates.append(dct)

    if (len(templates) > 0):
        templates = sorted(templates, key=lambda k: k['score'], reverse=True)

    return templates

def getTemplateRemoveFilters(f = template_rm_filters):
    return [key for key in f if (f[key])]



#     'Phone: first_character_type = ascii': False,
#
#     'Name: #ascii < 5': False,
#     'Name: _%digit = 0': False,
#     'Name: first_character_type != ascii': False

def checkTemplate(dct, dft):
    if (template_rm_filters['Phone: #digit < 8']):
        ndigit = sum([1 for c in dct['phone']['preprocessed'] if c in string.digits])
        if (ndigit < 8):
            return False

    if ('#ascii/(#ascii+#digit+#punctuation)' in dft and \
        '#digit/(#ascii+#digit+#punctuation)' in dft and \
        template_rm_filters['Phone: 2 * _%ascii < _%digit']):
        if (2 * dct['phone']['features'][dft['#ascii/(#ascii+#digit+#punctuation)']] >=
                dct['phone']['features'][dft['#digit/(#ascii+#digit+#punctuation)']]):
            return False

    if ('#ascii/(#ascii+#digit+#punctuation)' in dft and \
        '%kwPhone' in dft and \
         template_rm_filters['Phone: _%ascii > 0 & %kwPhone = 0']):
        if (dct['phone']['features'][dft['#ascii/(#ascii+#digit+#punctuation)']] > 0 and \
            dct['phone']['features'][dft['%kwPhone']] == 0):
            return False

    if (template_rm_filters['Phone: first_character_type != digit']):
        if (dct['phone']['preprocessed'] not in '(+' + string.digits):
            return False

    if ('#ascii' in dft and template_rm_filters['Name: #ascii < 5']):
        if (dct['name']['features'][dft['#ascii']] < 5):
            return False

    if ('#digit' in dft and template_rm_filters['Name: _%digit > 0']):
        if (dct['name']['features'][dft['#digit']] > 0):
            return False

    if ('first_character_ascii' in dft and template_rm_filters['Name: first_character_type != ascii']):
        if (dct['name']['features'][dft['first_character_ascii']] != 1):
            return False

    if (template_rm_filters['Name: _%ascii + _%digit < 1']):
        ndigit = sum([1 for c in dct['name']['preprocessed'] if c in string.digits])
        nascii = sum([1 for c in dct['name']['preprocessed'] if c in string.ascii_letters])
        if (nascii + ndigit < len(dct['name']['preprocessed'])):
            return False


    return True