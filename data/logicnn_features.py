"""
BUT-rule feature extractor

"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time

warnings.filterwarnings("ignore")   

def text_after_first(text, part):
    if part in text:
        return ''.join(text.split(part)[1:])
    else:
        return ''

def extract_but(revs):
    but_fea = []
    but_ind = []
    but_fea_cnt = 0
    for rev in revs:
        text = rev["text"]
        if ' but ' in text:
            but_ind.append(1)
            # make the text after 'but' as the feature
            fea = text.split('but')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        else:
            but_ind.append(0)
            fea = ''
        but_fea.append(fea)
    print '#but %d' % but_fea_cnt
    return {'but_text': but_fea, 'but_ind': but_ind}

if __name__=="__main__":
    data_file = sys.argv[1]
    print "loading data..."
    x = cPickle.load(open(data_file,"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    but_fea = extract_but(revs)
    cPickle.dump(but_fea, open("%s.fea.p" % data_file, "wb"))
    print "feature dumped!"

