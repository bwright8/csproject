import numpy, idx2numpy
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
X = idx2numpy.convert_from_file('train-images-idx3-ubyte')

X_test = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
y_test = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

Xp = []

for x in X:
    Xp.append(x.flatten())

n_train = len(y)

Xp = numpy.matrix(Xp)/255.0
Xp = numpy.hstack((Xp,numpy.ones((n_train,1))))

n_test = len(y_test)
Xpt = []
for x in X_test:
    Xpt.append(x.flatten())

Xpt = numpy.matrix(Xpt)/255.0
Xpt = numpy.hstack((Xpt,numpy.ones((n_test,1))))


"""
U,s,VT = numpy.linalg.svd(Xp,full_matrices = False)


XpTXp = VT.T @ (numpy.diag(s)) @ VT

print(XpTXp.shape)

XpTXpi = numpy.linalg.inv(XpTXp)
XpTXpiXT = XpTXpi@(Xp.T)
"""
#uncomment when needed

def linear_classifier_for_digit(d,XpTXpiXT,y):

    yp = []
    for i in y:
        if(i == d):
            yp.append(1)
        else:
            yp.append(-1)

    w = XpTXpiXT@yp


    return w

# w0 = linear_classifier_for_digit(0,XpTXpiT,y)
w = []
for i in range(10):
    #w.append(linear_classifier_for_digit(i,XpTXpiT,y))
    pass

# ridge classifier for parameter labmda

def ridge_classifier_for_digit(d,Xp,XpTXp,y,lam):
    yp = []
    for i in y:
        if(i == d):
            yp.append(1)
        else:
            yp.append(-1)


    XR = numpy.linalg.inv(XpTXp + lam*numpy.identity(784+1))
    w = XR@(Xp.T)@yp


    return w

#rw = ridge_classifier_for_digit(0,Xp,XpTXp,y,1)


def supportvector_classifier_for_digit(d,Xp,y):
    yp = []
    for i in y:
        if(i == d):
            yp.append(1)
        else:
            yp.append(-1)

    n_train = len(yp)
    X = Xp
    #X = Xp/255.0

    #xt = numpy.hstack((X,numpy.ones((n_train,1))))
    xt = Xp

    clf = LinearSVC(random_state=1, tol=.001,max_iter = 100000)
    clf.fit(xt, yp)
    w_opt = clf.coef_.transpose()

    return w_opt
    

"""from article https://scikit-learn.org/stable/modules/naive_bayes.html print("Number of mislabeled points out of a total %d points : %d"
...       % (X_test.shape[0], (y_test != y_pred).sum()))"""
