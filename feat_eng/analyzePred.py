import sys
from csv import DictReader, DictWriter
import numpy as np
from collections import defaultdict

def main(argv):
    try:
        f = open("train.csv", 'r')
        train = DictReader(f)
        
    except  IOError:
        print "Could Not Open File"
        exit(0)
    train_d = defaultdict(int)
    for ii in train:
        train_d[ii['id']] = ii['cat']
        #print ii['id']
    #print train_d
    f.close()
    try:
        f = open("predictions.csv", 'r')
        pred = DictReader(f)
    except  IOError:
        print "Could Not Open File"
        exit(0)
    pred_d = defaultdict(int)
    correct_cnt = 0.0
    for yy in pred:
        print yy['id']
        #print train_d[yy['id']]
        if train_d[yy['id']] == yy['cat']:
            correct_cnt += 1
    f.close()
    print train_d["112381"]
    print "accuracy: ", (correct_cnt/len(train_d)) * 100
        
if __name__ == "__main__":
    main(sys.argv)