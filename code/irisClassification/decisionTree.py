# -*- coding: utf-8 -*-#
'''
# Name:         decisionTree
# Description:  
# Author:       super
# Date:         2020/3/17
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    # 为了可视化，仅使用前两列特征
    x = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=24)
    return x_train, x_test, y_train, y_test

def show_depth_and_acc(x_train, y_train, x_test, y_test):
    # 看一下树不同深度模型的拟合效果
    depth = np.arange(1, 15)
    acc_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=d)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)
        acc_list.append(accuracy)
        print(d, ' 准确度: %.2f%%' % (100 * accuracy))
    plt.figure(facecolor='w')
    plt.plot(depth, acc_list, 'ro--', lw=2)
    plt.xlabel('depth of decision tree')
    plt.ylabel('acc')
    plt.title('depth and acc')
    plt.grid(True)
    plt.savefig('depthAndAcc.png')
    plt.show()

if __name__ == "__main__":
    path = 'iris.data'
    x_train, x_test, y_train, y_test = load_data(path)

    model = Pipeline([
        ('ss', StandardScaler()),
        ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3))
    ])
    model.fit(x_train, y_train)
    print('准确度(score函数): %.2f%%' % (100 * model.score(x_test, y_test)))

    y_test_hat = model.predict(x_test).reshape(-1, 1)
    y_test = y_test.reshape(-1,1)
    result = (y_test_hat == y_test)
    acc = np.mean(result)
    print('准确度(手动计算): %.2f%%' % (100 * acc))

    # 保存决策树
    f = open('./iris_tree.dot', 'w')
    tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f)

    show_depth_and_acc(x_train, y_train, x_test, y_test)