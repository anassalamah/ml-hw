import argparse
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

class Analyzer:
    def __init__(self, word, before, after, prev, next, char, dict):
        self.word = word
        self.after = after
        self.before = before
        self.prev = prev
        self.next = next
        #self.dict = dict
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
        #if self.dict:
            #for ii in [x for x in feats if x.startswith("D:")]:
                #yield ii
        if self.char:
            for ii in [x for x in feats if x.startswith("C:")]:
                yield ii


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        print "EXAMPLE: ", examples
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    x_train = list(DictReader(open("train.csv", 'r')))
    x_test = x_train
    
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
    
    #feat = Featurizer(analyzer=analyzer)
    feat = Featurizer()
    
    labels = []
    for line in x_train:
        if not line['cat'] in labels:
            labels.append(line['cat'])

    x_train = feat.train_feature(x['text'] for x in x_train)
    x_test = feat.test_feature(x['text'] for x in x_test)

    y_train = array(list(labels.index(x['cat']) for x in x_train))
    print "Y_TRAIN", y_train
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    #feat.show_top10(lr, labels)
    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train, all_examples(flags.limit))
    print("TEST\n-------------------------")
    accuracy(lr, x_test, y_test, all_examples(flags.limit, train=False))
    
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in x_test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
