import numpy, idx2numpy
from sklearn.svm import LinearSVC

y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
X = idx2numpy.convert_from_file('train-images-idx3-ubyte')


Xp = []

for x in X:
    Xp.append(x.flatten())


Xp = numpy.matrix(Xp)/256.0


U,s,VT = numpy.linalg.svd(Xp,full_matrices = False)


XpTXp = VT.T @ (numpy.diag(s)) @ VT

print(XpTXp.shape)

XpTXpi = numpy.linalg.inv(XpTXp)
XpTXpiT = XpTXpi@(Xp.T)


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


    XR = numpy.linalg.inv(XpTXp + lam*numpy.identity(784))
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
    #X = Xp/256.0

    xt = numpy.hstack((X,numpy.ones((n_train,1))))

    clf = LinearSVC(random_state=1, tol=.001,max_iter = 100000)
    clf.fit(xt, yp)
    w_opt = clf.coef_.transpose()

    return w_opt
    
