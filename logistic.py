# -*- coding:utf-8 -*- 
import sys
import re
import io
import random
import csv
import math
import pandas
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.integrate import quad
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression


class Logistic(object):

    def __init__(self):
        pass

    def input_text(self):                                            
        names = ["ok", "ra", "age", "education", "employees", "companies"]
        frame = pandas.read_csv("text/calbee",sep = ",", header = None, names = names)
        return frame

    def input_predict(self):                                            
        names = ["ok", "ra", "age", "education", "employees", "companies", "model"]
        predict = pandas.read_csv("text/predict",sep = ",", header = None, names = names)
        predict = logistic.cleansing(predict)
        predict = logistic.add_variable(predict)
        predict.columns= ["index","ok","ra","age","education","employees","companies","model", "age_zscore","education_zscore","companies_zscore","edu1","edu2","edu3","edu4","edu5","edu6"]
        predict["edu7"] = 0.
        return predict

    def cleansing(self, frame):
        #object to float
        frame.ok = frame.ok.convert_objects(convert_numeric=True)
        #NaNレコード削除
        frame = frame.dropna()
        #index振り直し
        frame = frame.reset_index()
        return frame

    def add_variable(self, frame):

        #add_zscore
        frame["age_zscore"]= stats.zscore(frame.age)
        frame["education_zscore"]= stats.zscore(frame.education)
        frame["companies_zscore"]= stats.zscore(frame.companies)

        #add_education_dummy
        dummy = pandas.get_dummies(frame.education)
        frame = pandas.merge(frame, dummy, right_index=True, left_index=True) #index結合

        return frame

    def hist_show(self, frame, feature):

        #hist_args
        xmax = frame[feature].max()
        xmin = frame[feature].min()
        bins = int(math.ceil(xmax) - math.floor(xmin) + 2)
        range = (math.floor(xmin) - 1, math.ceil(xmax) + 1)

        plt.figure()
        plt.hist(frame[feature]                             , alpha = .4, bins = bins, range = range, color = "g", label = "CA")
        plt.hist(frame[frame.ra == 1].reset_index()[feature], alpha = .4, bins = bins, range = range, color = 'b', label = "RA")
        plt.hist(frame[frame.ok == 1].reset_index()[feature], alpha = .4, bins = bins, range = range, color = 'r', label = "OK")

        #mean_line
        plt.axvline(x = frame[feature].mean()               , linewidth = 1, color = 'g')
        plt.axvline(x = frame[frame.ra == 1][feature].mean(), linewidth = 1, color = 'b')
        plt.axvline(x = frame[frame.ok == 1][feature].mean(), linewidth = 1, color = 'r')

        plt.legend()
        plt.show()

    def logistic_fit(self, cls, features):

        classifier = LogisticRegression()
        classifier.fit(features, cls)
        print(features.columns)
        print(classifier.coef_)
        print(classifier.intercept_)
        print(classifier.score(features, cls))
        return classifier

    def logistic_predict(self, classifier, features):
                                            
        probas_ = classifier.predict_proba(features)
        list_ = classifier.predict(features)
        return [probas_, list_]
        # 1. / (1. + np.exp(-prob))

        #roc_show
        """
        roc_auc = auc(fpr, tpr)
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
        """

    def gain_show(self):

        rocframe = pandas.read_csv("text/roc",sep="\t", header=None, names=["age"])
        print(rocframe)
        plt.figure()
        plt.plot(rocframe)
        plt.show()

logistic = Logistic()
frame = logistic.input_text()
frame = logistic.cleansing(frame)
frame = logistic.add_variable(frame)

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
