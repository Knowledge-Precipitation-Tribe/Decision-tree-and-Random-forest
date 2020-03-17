# -*- coding: utf-8 -*-#
'''
# Name:         decisionTreeRegressor
# Description:  
# Author:       super
# Date:         2020/3/17
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def load_data():
    N = 100
    x = np.random.rand(N) * 6 - 3  # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    x = x.reshape(-1, 1)
    return x, y

if __name__ == "__main__":
    x, y = load_data()
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    depth = [2,4,6,8,10]
    clr = 'rgbmy'

    plt.plot(x, y, 'k^', linewidth=2, label='Actual')
    for i, d in enumerate(depth):
        reg = DecisionTreeRegressor(criterion='mse', max_depth=d)
        reg.fit(x,y)
        y_hat = reg.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=clr[i], linewidth=2, label='Depth=%d' % d)
    plt.legend()
    plt.savefig('DecisionTreeRegressor.png')
    plt.show()