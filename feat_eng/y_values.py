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
    
    labels = defaultdict(int)
    for line in train:
        labels[line['cat']] += 1
    tot = sum(labels.values())
    for i in labels.keys():
        labels[i]=float(labels[i])/ tot
    for i in sorted(labels, key=labels.get, reverse=True):
        print i, labels[i]
    
if __name__ == "__main__":
    main(sys.argv)