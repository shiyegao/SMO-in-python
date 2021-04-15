import sys
import numpy as np
import pandas as pd
from sklearn import svm
#from math import *

class SMO():
    def __init__(self, C=0.6, toler=0.001, max_iter=40, mv_tol=0.1):
        self.C = C
        self.toler = toler
        self.maxIter = max_iter
        self.mv_tol = mv_tol

    def selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0,m))
        return j

    def clipAlpha(self, a_j, H, L):
        if a_j > H: a_j = H
        if L > a_j: a_j = L
        return a_j

    def fit(self, data, label):
        dataMatrix = np.mat(data)
        label = np.array([-1 if i==0 else 1 for i in label])
        labelMatrix = np.mat(label).transpose()
        # print(dataMatrix.shape, labelMatrix.shape)
        b = 0.0
        iter = 0
        m = np.shape(dataMatrix)[0]
        alpha = np.mat(np.zeros((m,1)))
        while iter < self.maxIter:
            alphapairChanged = 0
            for i in range(m):
                fxi = float(np.multiply(alpha, labelMatrix).T * (dataMatrix * dataMatrix[i,:].T)) + b
                Ei = fxi - float(labelMatrix[i])
                # if Ei!=0: print(Ei, "yesssssssssss")
                if labelMatrix[i] * Ei < -self.toler and alpha[i] < self.C or labelMatrix[i] * Ei > self.toler and alpha[i] > 0:
                    # print("yes")
                    j = self.selectJrand(i,m)
                    fxj = float(np.multiply(alpha,labelMatrix).T * (dataMatrix * dataMatrix[j,:].T)) + b
                    Ej = fxj - float(labelMatrix[j])
                    alphaIOld = alpha[i].copy()
                    alphaJOld = alpha[j].copy()
                    if labelMatrix[i] != labelMatrix[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C+alpha[j]-alpha[i])
                    else:
                        L = max(0, alpha[i]+alpha[j] - self.C)
                        H = min(self.C, alpha[j]+alpha[i])
                    if L==H:
                        # print("L==H")
                        continue
                    eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                    if eta >= 0:
                        print("eta >= 0")
                        continue
                    alpha[j] -= labelMatrix[j]*(Ei-Ej)/eta
                    alpha[j] = self.clipAlpha(alpha[j],H,L)
                    if abs(alpha[j] - alphaJOld) < self.mv_tol :
                        # print("j not move enough, no need for updating")
                        continue
                    alpha[i] += labelMatrix[j]*labelMatrix[i]*(alphaJOld - alpha[j])
                    b1 = b - Ei -labelMatrix[i]*(alpha[i] - alphaIOld)*dataMatrix[i,:]*dataMatrix[i,:].T \
                        -labelMatrix[j]*(alpha[j]-alphaJOld)*dataMatrix[i,:]*dataMatrix[j,:].T
                    b2 = b - Ej -labelMatrix[i]*(alpha[i] - alphaIOld)*dataMatrix[i,:]*dataMatrix[j,:].T \
                        -labelMatrix[j]*(alpha[j]-alphaJOld)*dataMatrix[j,:]*dataMatrix[j,:].T
                    if alpha[i] > 0 and alpha[i] < self.C: b = b1
                    elif alpha[j] > 0 and alpha[j] < self.C: b = b2
                    else: b = (b1+b2)/2.0
                    alphapairChanged +=1
                    # print("iter: %d i:%d, alphas changed %d" %(iter,i,alphapairChanged))
            if alphapairChanged == 0:
                iter += 1
            else:
                iter = 0
            if not iter%10 and iter!=0: print("iteration number: %d" % iter)
        self.b = b
        self.alpha = alpha
        self.w = dataMatrix.T @ np.multiply(alpha, labelMatrix)
        np.savez("parameter_0001.npz", b=b, alpha=alpha, w=self.w)
        return b, alpha, self.w

    def test(self, data, label, w=None):
        if w is None: w=self.w
        dataMatrix = np.mat(data)
        labelMatrix = np.mat(label).transpose()
        res = (dataMatrix @ w) > 0
        return np.sum(res==labelMatrix)/labelMatrix.shape[0]

def make_datasets():    
    data = pd.read_csv('./datasets/gpl96.csv')
    d = np.array(data.values.tolist())
    x_train = d[:, :-1]
    y_train = d[:, -1]
    print(x_train.shape)
    print(y_train.shape)
    

    data = pd.read_csv('./datasets/gpl97.csv')
    d = np.array(data.values.tolist())
    x_test = d[:, :-1]
    y_test = d[:, -1]
    print(x_test.shape)
    print(y_test.shape)

    np.savez("data.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

def main():

    # make_datasets()

    # loading
    data = np.load("data.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Sklearn
    sk_model = svm.LinearSVC(C=0.6, tol=0.001)
    sk_model.fit(x_train, y_train)
    sk_acc = sk_model.score(x_test, y_test)
    print("Sklearn Accuracy:", sk_acc)

    # training
    model = SMO(C=0.6, toler=0.001, max_iter=40, mv_tol=0.001)
    # b, _, _ = model.fit(x_train,y_train)
    # print(b)

    # testing
    w = np.load("parameter_0001.npz")["w"]
    acc = model.test(x_test, y_test, w)
    print("My Accuracy:", acc)



if __name__ == '__main__':
    sys.exit(main())