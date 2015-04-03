# -*- coding:utf-8 -*- 
import sys
import re
import io
import numpy
import random
import csv
import math
import pandas
from matplotlib import pyplot as plt
from scipy.integrate import quad
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression


class Zpoint(object):

    def __init__(self):
        pass

    def input_text(self):
        
        frame = pandas.read_csv("text/calbee",sep=",", header=None, names=["cls", "age", "level", "span"])
        return frame

    def gain_show(self):

        rocframe = pandas.read_csv("text/roc",sep="\t", header=None, names=["age"])
        print(rocframe)
        plt.figure()
        plt.plot(rocframe)
        plt.show()

    def hist_show(self, feature):

        plt.figure()   
        plt.hist(feature, alpha = 0.3, normed = False, histtype = 'stepfilled', color = 'r')
        plt.show()

    def zpoint_show(self, feature):

         average = numpy.average(feature)
         deviation = numpy.std(feature)
         frame = pandas.DataFrame(feature)
         frame['zpoint'] = (feature - average)/deviation
         print('平均値:%f 標準偏差:%f' %(average, deviation))
         print(frame)

    def logistic(self, frame):
        #目的変数
        cls = frame.cls
        #説明変数
        features = frame[['age','level']]
        #fitting
        classifier = LogisticRegression()
        classifier.fit(features,cls)
        #predict
        probas_ = classifier.predict_proba(features)
        #roc
        fpr, tpr, thresholds = roc_curve(cls, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        #show
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

zpoint = Zpoint()
frame = zpoint.input_text()

'''確率　
end = float("-inf")
for x in sample:
 answer, abserr = quad(lambda x:(1/(math.sqrt(2*math.pi)*deviation))*math.exp(-pow((x-average),2.0)/(2*pow(deviation,2.0))),end,x)
 print("x:%f , answer:%f" %(x, answer))
'''
'''
f = open("./text/roc")
text = f.read()
roclist = text[:-1].split('\n')
roclist = list(map(int, roclist))
'''
