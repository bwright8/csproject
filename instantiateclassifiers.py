import numpy, idx2numpy
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from learnermodv2 import *


"""
gnb = GaussianNB()
nb_classifier = gnb.fit(Xp,y)

"""

lam = [100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0]

def msvc():

    for i in range(10):
        wsv = supportvector_classifier_for_digit(i,Xp,y,l =.1)
        numpy.savetxt("wsvrr"+str(i)+".csv",wsv)



def mlrc():
	
    for i in range(10):
        lr = ridge_classifier_for_digit(i,Xp,XpTXp,y,lam[i])
        numpy.savetxt("lrr"+str(i)+".csv",lr)
    

