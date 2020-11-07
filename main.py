import numpy, idx2numpy

y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
X = idx2numpy.convert_from_file('train-images-idx3-ubyte')


Xp = []

for x in X:
    Xp.append(x.flatten())


Xp = numpy.matrix(Xp)


U,s,VT = numpy.linalg.svd(Xp,full_matrices = False)


XpTXp = VT.T @ (numpy.diag(s)) @ VT

print(XpTXp.shape)

XpTXpi = numpy.linalg.inv(XpTXp)
XpTXpiT = XpTXpi@(Xp.T)


def classifier_for_digit(d,XpTXpiXT,y):

    yp = []
    for i in y:
        if(i == d):
            yp.append(1)
        else:
            yp.append(-1)

    w = XpTXpiXT@yp


    return w

w0 = classifier_for_digit(0,XpTXpiT,y)
