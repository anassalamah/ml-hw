"""
Feature Engineering
Anas Salamah
"""

import argparse
import string
from collections import defaultdict
import operator

from csv import DictReader, DictWriter
from numpy import array

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score

from nltk.corpus import wordnet as wn, stopwords as sw
from nltk.corpus import brown
from nltk.util import ngrams
import nltk


def normalize_tags(tag):
    if not tag or not tag[0] in string.uppercase:
        return "PUNC"
    else:
        return tag[:2]


kTAGSET = ["", "Literature", "History", "Social Studies", "Fine Arts", "Other", "Physics", "Social Science", "Biology", "science", "Chemistry", "Mathematics", "Geography", "Astronomy", "earth Science"]


class Analyzer:
    def __init__(self, words, sw, length, bigram, cnt, wordcnt):
        self.words = words
        self.length = length
        self.sw = sw
        self.bigram = bigram
        self.cnt = cnt
        self.wordcnt = wordcnt

    def __call__(self, feature_string):
        feats = feature_string.split()
        #print "FEATS: " ,feature_string
        #print "FEETSSPLIT: ", feats
        if self.words:
            for ii in [x for x in feats if x.startswith("W:")]:
                yield ii

        if self.length:
            for ii in [x for x in feats if x.startswith("L:")]:
                #print "after", ii
                yield ii
        if self.sw:
            for ii in [x for x in feats if x.startswith("STOP:")]:
                yield ii
        if self.bigram:
            for ii in [x for x in feats if x.startswith("BIGRAM:")]:
                yield ii
        if self.cnt:
            for ii in [x for x in feats if x.startswith("N:")]:
                yield ii
            for ii in [x for x in feats if x.startswith("ADV:")]:
                yield ii
            for ii in [x for x in feats if x.startswith("VBD:")]:
                yield ii
        if self.wordcnt:
            for ii in [x for x in feats if x.startswith("CNT")]:
                yield ii
        
def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

def example(sentence, ct):
        ex = ""
        vbd = 0
        adv = 0
        n = 0
        d = defaultdict(int)
        #pos = nltk.pos_tag(sentence)
        #print "POS::::", pos
        stop = sw.words("english")
        for i in xrange(len(sentence)):
            stem = morphy_stem(sentence[i])
            if stem in stop:
                ex += " STOP:%s" % stem
            else:
                d[stem] += 1
                ex += " W:%s" % stem
            if sentence[i][-2:] == 'ed':
                vbd += 1
            elif sentence[i][-2:] == 'ly':
                adv += 1
            elif sentence[i][0].isupper():
                n += 1
            if i != len(sentence)-1:
                stem2 = morphy_stem(sentence[i+1])
                if not stem2 in stop:
                    ex += " BIGRAM:%s%s" % (stem,stem2)
            
            #ex += i
        if ct in kTAGSET:
            target = kTAGSET.index(ct)
            #ex += " CT:%s" % ct
            #ex += ct
        else:
            target = None
        for i in d:
            ex += " CNT"+i+":%i" % d[i]
        ex += " VBD:%i" % vbd
        ex += " ADV:%i" % adv
        ex += " N:%i" % n
        ex += " L:%s" % str(len(ex.split()))
        """if position > 0:
            prev = " P:%s" % sentence[position - 1]
        else:
            prev = ""

        if position < len(sentence) - 1:
            next = " N:%s" % sentence[position + 1]
        else:
            next = ''

        all_before = " " + " ".join(["B:%s" % x
                                     for x in sentence[:position]])
        #print "HHHHH", all_before
        all_after = " " + " ".join(["A:%s" % x
                                    for x in sentence[(position + 1):]])
        """
        """
        char = ' '
        padded_word = "~%s^" % sentence[position]
        for ngram_length in xrange(2, 5):
            char += ' ' + " ".join("C:%s" % "".join(cc for cc in x)
                                   for x in ngrams(padded_word, ngram_length))
        ex += char

        ex += prev
        ex += next
        ex += all_after
        ex += all_before
        #print "EXAMPLE", ex
        #ex += dictionary
        """
        
        return ex, ct


def all_examples(limit, train=True):
    ex_num = 0
    if train:
        data_x = list(DictReader(open("train.csv", 'r')))
    else:
        data_x = list(DictReader(open("test.csv", 'r')))
    for ii in data_x:
        ex_num += 1
        #if train:
           #print "TRAIN:", ex_num
        #else:
            #print "TEST: ", ex_num
        if limit > 0 and ex_num > limit:
            break
        #print "EX: ", ii['text']
        
        if train:
            ex,ct = example(ii['text'].split(), ii['cat'])
            yield ex, ct
        else:
            ex,ct = example(ii['text'].split(), "NONE")
            yield ex

        
def accuracy(classifier, x, y, examples):
    predictions = classifier.predict(x)
    cm = confusion_matrix(y, predictions)

    print("Accuracy: %f" % accuracy_score(y, predictions))

    print("\t".join(kTAGSET[1:]))
    for ii in cm:
        print("\t".join(str(x) for x in ii))

    errors = defaultdict(int)
    for ii, ex_tuple in enumerate(examples):
        ex, tgt = ex_tuple
        #print "HELLO: ", ex, tgt
        #print "EEEE", predictions[ii]
        if tgt != predictions[ii]:
            errors[(ex, predictions[ii])] += 1

    for ww, cc in sorted(errors.items(), key=operator.itemgetter(1),
                         reverse=True)[:10]:
        print("%s\t%i HIIIIII" % (ww, cc))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--words', default=False, action='store_true',
                        help="Use words as features")
    parser.add_argument('--sw', default=False, action='store_true',
                        help="Use stopwords as features")
    parser.add_argument('--length', default=False, action='store_true',
                        help="Use length as features")
    parser.add_argument('--bigram', default=False, action='store_true',
                        help="Use bigrams as feature")
    parser.add_argument('--cnt', default=False, action='store_true',
                        help="Use N, VBD, ADV counts as feature")
    parser.add_argument('--wordcnt', default=False, action='store_true',
                        help="Use wordcnt features")
    parser.add_argument('--limit', default=-1, type=int,
                        help="How many sentences to use")

    flags = parser.parse_args()

    analyzer = Analyzer(flags.words, flags.sw, flags.length,
                        flags.bigram, flags.cnt, flags.wordcnt)
    vectorizer = HashingVectorizer(analyzer=analyzer)

    x_train = vectorizer.fit_transform(ex for ex, tgt in
                                       all_examples(flags.limit))
    x_test = vectorizer.fit_transform(ex for ex in
                                      all_examples(flags.limit, train=False))
    #print "FFF", x_train['id']
    for ex, tgt in all_examples(1):
        print(" ".join(analyzer(ex)))

    y_train = array(list(tgt for ex, tgt in all_examples(flags.limit)))
    #print "Y_TRAIN", y_train
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train, all_examples(flags.limit))
    
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in list(DictReader(open("test.csv", 'r')))], predictions):
        #print "PP", pp
        d = {'id': ii, 'cat': pp}
        o.writerow(d)
