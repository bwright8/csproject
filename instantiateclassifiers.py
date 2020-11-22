import numpy, idx2numpy
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from learnermodv2 import *


"""
gnb = GaussianNB()
nb_classifier = gnb.fit(Xp,y)

"""

def msvc():
    for i in range(10):
        wsv = supportvector_classifier_for_digit(i,Xp,y)
        numpy.savetxt("wsv"+str(i)+".csv",wsv)



def mlrc():
    for i in range(10):
        lr = linear_classifier_for_digit(i,XpTXpiXT,y)
        numpy.savetxt("lr"+str(i)+".csv",lr)
    

