# -*- coding: utf-8 -*-#
'''
# Name:         decisionTree
# Description:  
# Author:       super
# Date:         2020/3/13
'''
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import display, Image

def load_data(path):
    data = pd.read_csv(path)
    # print(data.shape)
    # print(data.head())
    data['好瓜'] = pd.factorize(data['好瓜'])[0].astype(np.uint16)
    data['色泽'] = pd.factorize(data['色泽'])[0].astype(np.uint16)
    data['根蒂'] = pd.factorize(data['根蒂'])[0].astype(np.uint16)
    data['敲声'] = pd.factorize(data['敲声'])[0].astype(np.uint16)
    data['纹理'] = pd.factorize(data['纹理'])[0].astype(np.uint16)
    data['脐部'] = pd.factorize(data['脐部'])[0].astype(np.uint16)
    data['触感'] = pd.factorize(data['触感'])[0].astype(np.uint16)
    y = data['好瓜']
    x = data.drop(['编号','好瓜'], axis = 1)
    return x, y

def build_model():
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    return clf

if __name__ == '__main__':
    x, y = load_data("data.csv")
    # print(x)
    # print(y)
    clf = build_model()
    clf.fit(x.values, y.values)
    print(clf)

    dot_data = export_graphviz(clf,
                               out_file="tree.dot",
                               feature_names=x.columns,
                               filled=True,
                               rounded=True)