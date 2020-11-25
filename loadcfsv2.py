import numpy
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from learnermodv2 import *
from joblib import dump,load

gnb = GaussianNB()
nb_classifier = gnb.fit(Xp,y)


wsvcs = []
def lsvc():
    for i in range(10):
        wsvcs.append(numpy.loadtxt("wsv"+str(i)+".csv"))
lsvc()


wlrcs = []
def llrc():
    for i in range(10):
        wlrcs.append(numpy.loadtxt("lr"+str(i)+".csv"))
llrc()

def digit_selector(datapoint,digitclassifiers):
    nmax = -10000
    kmax = 0

    for i in range(10):
        p = datapoint@digitclassifiers[i]

        if(p > nmax):
            nmax = p
            kmax = i
    return(kmax)
    
    

wrongsv = 0
wronglr = 0
wrongnb = 0
CAP = n_train
#CAP = 5000
clf = load('nn.joblib')
"""
for i in range(n_test):
    if(digit_selector(Xpt[i],wlrcs) != y_test[i]):
        wronglr += 1
    if(digit_selector(Xpt[i],wsvcs) != y_test[i]):
        wrongsv += 1
    if(nb_classifier.predict(Xpt[i])!= y_test[i]):
        wrongnb += 1
"""
yp = clf.predict(Xpt)
print(accuracy_score(y_test,yp))
