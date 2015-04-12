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


class Model(object):

    def __init__(self):
        pass

    def input_train(self):                                            
        names = ["ok", "ra", "age", "education", "employees", "companies"]
        frame = pandas.read_csv("text/calbee",sep = ",", header = None, names = names)
        return frame

    def input_test(self, frame):                                            
        names = ["ok", "ra", "age", "education", "employees", "companies", "model"]
        predict = pandas.read_csv("text/predict",sep = ",", header = None, names = names)
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

    def add_variable2(self, frame, predict):

        #frame
        age_ave = frame.age.mean()
        age_std = frame.age.std()
        edu_ave = frame.education.mean()
        edu_std = frame.education.std()
        com_ave = frame.companies.mean()
        com_std = frame.companies.std()
        #add_zscore
        predict["age_zscore"] =  ( predict.age - age_ave ) / age_std
        predict["education_zscore"] =  ( predict.education - edu_ave ) / edu_std
        predict["companies_zscore"] =  ( predict.companies - com_ave ) / com_std

        #add_education_dummy
        dummy = pandas.get_dummies(predict.education)
        predict = pandas.merge(predict, dummy, right_index=True, left_index=True) #index結合

        return predict

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

'''statsmodel　
import statsmodels.api as sm
train_cols=["age_zscore","education_zscore","companies_zscore"]
logit = sm.Logit(frame['ok'], frame[train_cols])
result_n = logit.fit()
result_n.summary()
'''

'''
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
'''
