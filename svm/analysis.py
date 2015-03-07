import argparse
import numpy as np
from csv import DictWriter
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        
        train_x, train_y = train_set
        self.train_x=[]
        self.train_y=[]
        for i in xrange(len(train_y)):
            if train_y[i] in set([3,8]):
                self.train_x.append(train_x[i])
                self.train_y.append(train_y[i])
        test_x, test_y = valid_set
        self.test_x=[]
        self.test_y=[]
        for i in xrange(len(test_y)):
            if test_y[i] in set([3,8]):
                self.test_x.append(train_x[i])
                self.test_y.append(train_y[i])
                
        self.test_x=array(self.train_x)
        self.test_y=array(self.train_y)
        self.test_x=array(self.test_x)
        self.test_y=array(self.test_y)
        f.close()

def plot_model(X, Y, clf, fignum=0):
    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = min(X[:, 0])
    x_max = max(X[:, 0])
    y_min = min(X[:, 1])
    y_max = max(X[:, 1])

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM Analysis Section')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    # You should not have to modify any of this code
    fignum = 0
    data = Numbers("../data/mnist.pkl.gz")

    output = DictWriter(open('result1.csv', 'w'), ['Accuracy','Kernel','C','Degree','Gamma','Coefficient'], lineterminator='\n')
    output.writeheader()
    
    if args.limit > 0:
        #print "train_x: ", data.train_x[:args.limit]
        #print "train_y: ", data.train_y[::args.limit]
        
        for c, kk, dd, gg, cc in [
                                  
                           (1, 'linear', 0, 0, 0)]:
        # trail 1
        #kk = 'poly'
        #dd = 1
        #gg = 1
        #cc = 0
            clf = svm.SVC(C=c, kernel=kk, degree=dd, coef0=cc, gamma=gg)
            clf.fit(data.train_x[:args.limit], data.train_y[:args.limit])
            pred = clf.predict(data.test_x[:args.limit])
            actual = data.test_y[:args.limit]
            predlen = len(pred)
            correct = 0.0
           
            for i in xrange(predlen):
                if pred[i] == actual[i]:
                    correct += 1
            output.writerow({"Accuracy" : correct/predlen, "Kernel" : kk, "C" : c, 'Degree' : dd, 'Gamma' : gg, 'Coefficient' : cc })
            #plot_model(data.train_x[:args.limit], data.train_y[:args.limit], clf, fignum)
            #fignum += 1
    else:
        for c, kk, dd, gg, cc in [(1, 'linear', 0, 0, 0)
                           ]:
            # Fit the model
            clf = svm.SVC(kernel=kk, degree=dd, coef0=cc, gamma=gg)
            clf.fit(data.train_x, data.train_y)
            
            pred = clf.predict(data.test_x)
            actual = data.test_y
            predlen = len(pred)
            correct = 0.0
           
            for i in xrange(predlen):
                if pred[i] == actual[i]:
                    correct += 1
            print "writing", kk, cc
            output.writerow({"Accuracy" : correct/predlen, "Kernel" : kk, "C" : c, 'Degree' : dd, 'Gamma' : gg, 'Coefficient' : cc })
            
            #plot_model(data.train_x, data.train_y, clf, fignum)
            #fignum += 1
    print "n support", clf.n_support_
    plt.imshow(clf.support_vectors_.reshape((28, 28)), cmap = cm.Greys_r)
    #plt.imshow(data.train_x[0].reshape((28, 28)), cmap = cm.Greys_r)
    plt.show()
    # Create File for predictions
    
    
       

    
