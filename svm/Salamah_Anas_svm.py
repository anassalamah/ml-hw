from numpy import array, zeros

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """
    #print "x: ", x
    #print "y: ", y
    #print "alpha: ", alpha
    w = zeros(len(x[0]))
    # TODO: IMPLEMENT THIS FUNCTION
    
    for i in xrange(len(x)):
        w[0] += alpha[i]*y[i]*x[i][0]
        w[1] += alpha[i]*y[i]*x[i][1]
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """

    support = set()
    #print "x: ", x
    #print "y: ", y
    #print "w: ", w
    #print "b: ", b
    # TODO: IMPLEMENT THIS FUNCTION
    for i in xrange(len(x)):
        if (y[i]*((w[0]*x[i][0]+w[1]*x[i][1])+b)) - 1.0 <= tolerance:
            support.add(i)
    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """
    #print "x: ", x
    #print "y: ", y
    #print "w: ", w
    #print "b: ", b
    slack = set()
    # TODO: IMPLEMENT THIS FUNCTION
    for i in xrange(len(x)):
        if (y[i]*((w[0]*x[i][0]+w[1]*x[i][1])+b)) < 1:
            slack.add(i)
    return slack
