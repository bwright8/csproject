import numpy, idx2numpy
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
X = idx2numpy.convert_from_file('train-images-idx3-ubyte')

from learnermodv2 import *

