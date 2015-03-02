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

from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.util import ngrams


def normalize_tags(tag):
    if not tag or not tag[0] in string.uppercase:
        return "PUNC"
    else:
        return tag[:2]


kTAGSET = ["", "Literature", "History", "Social Studies", "Fine Arts", "Other", "Physics", "Social Science", "Biology", "science", "Chemistry", "Mathematics", "Geography", "Astronomy", "earth Science"]


class Analyzer:
    def __init__(self, word, before, after, prev, next, char, dict):
        self.word = word
        self.after = after
        self.before = before
        self.prev = prev
        self.next = next
        self.dict = dict
        self.char = char

    def __call__(self, feature_string):
        feats = feature_string.split()

        if self.word:
            yield feats[0]

        if self.after:
            for ii in [x for x in feats if x.startswith("A:")]:
                yield ii
        if self.before:
            for ii in [x for x in feats if x.startswith("B:")]:
                yield ii
        if self.prev:
            for ii in [x for x in feats if x.startswith("P:")]:
                yield ii
        if self.next:
            for ii in [x for x in feats if x.startswith("N:")]:
                yield ii
        if self.dict:
            for ii in [x for x in feats if x.startswith("D:")]:
                yield ii
        if self.char:
            for ii in [x for x in feats if x.startswith("C:")]:
                yield ii


def example(sentence, position):
        word = sentence[position][0]
        ex = word
    

        if position > 0:
            prev = " P:%s" % sentence[position - 1][0]
        else:
            prev = ""

        if position < len(sentence) - 1:
            next = " N:%s" % sentence[position + 1][0]
        else:
            next = ''

        all_before = " " + " ".join(["B:%s" % x[0]
                                     for x in sentence[:position]])
        all_after = " " + " ".join(["A:%s" % x[0]
                                    for x in sentence[(position + 1):]])


        char = ' '
        padded_word = "~%s^" % sentence[position][0]
        for ngram_length in xrange(2, 5):
            char += ' ' + " ".join("C:%s" % "".join(cc for cc in x)
                                   for x in ngrams(padded_word, ngram_length))
        ex += char

        ex += prev
        ex += next
        ex += all_after
        ex += all_before
        ex += dictionary

        return ex


def all_examples(limit, train=True):
    sent_num = 0
    if train:
        data_x = list(DictReader(open("train.csv", 'r')))
    else:
        data_x = list(DictReader(open("test.csv", 'r')))
    for ii in data_x:
        sent_num += 1
        if limit > 0 and sent_num > limit:
            break
        elif train:
            yield ii['text'], ii['cat']
        else:
            yield ii['text']

        
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
        print("%s\t%i" % (ww, cc))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--word', default=False, action='store_true',
                        help="Use word features")
    parser.add_argument('--all_before', default=False, action='store_true',
                        help="Use all words before context as features")
    parser.add_argument('--all_after', default=False, action='store_true',
                        help="Use all words after context as features")
    parser.add_argument('--one_before', default=False, action='store_true',
                        help="Use one word before context as feature")
    parser.add_argument('--one_after', default=False, action='store_true',
                        help="Use one word after context as feature")
    parser.add_argument('--characters', default=False, action='store_true',
                        help="Use character features")
    parser.add_argument('--dictionary', default=False, action='store_true',
                        help="Use dictionary features")
    parser.add_argument('--limit', default=-1, type=int,
                        help="How many sentences to use")

    flags = parser.parse_args()

    analyzer = Analyzer(flags.word, flags.all_before, flags.all_after,
                        flags.one_before, flags.one_after, flags.characters,
                        flags.dictionary)
    vectorizer = HashingVectorizer(analyzer=analyzer)

    x_train = vectorizer.fit_transform(ex for ex, tgt in
                                       all_examples(flags.limit))
    x_test = vectorizer.fit_transform(ex for ex in
                                      all_examples(flags.limit, train=False))
    #print "FFF", x_train['id']
    for ex, tgt in all_examples(1):
        print(" ".join(analyzer(ex)))

    y_train = array(list(tgt for ex, tgt in all_examples(flags.limit)))
    print "Y_TRAIN", y_train
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train, all_examples(flags.limit))
    
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in list(DictReader(open("test.csv", 'r')))], predictions):
        print "PP", pp
        d = {'id': ii, 'cat': pp}
        o.writerow(d)
