import numpy, idx2numpy
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB


"""
gnb = GaussianNB()
nb_classifier = gnb.fit(Xp,y)

"""
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
    

