import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plots(x, y, b):
    
    plt.scatter(x,y)
    plt.plot(x,b[0]+b[1]*x)

#liner regression using exact minimum square root error
def linear_regression_exact(x, y):
    n = float(len(x))
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    SS_xy = sum(x*y) - n*x_mean*y_mean
    SS_xx = sum(x*x) - n*x_mean*x_mean
    
    b1 = SS_xy/SS_xx
    b0 = y_mean - b1*x_mean
    return b0, b1

#liner regression using gradient descent
def linear_regression_gd(x, y, b0=0, b1=0, iterations=1000, learning_rate=0.0001): 
    n = float(len(x))
    for i in range(iterations):
        y_current = b0 + b1*x
        b0_gradient = -(2/n) * sum(y - y_current)
        b1_gradient = -(2/n) * sum(x* (y - y_current))
        b0 = b0 - b0_gradient*learning_rate
        b1 = b1 - b1_gradient*learning_rate
    return b0, b1

#liner regression using scikit-learn
def linear_regression_scikit(x,y):
    model = LinearRegression(fit_intercept=True)
    X = x[:, np.newaxis]  #Why? https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
    model.fit(X, y)
    b0 = model.intercept_
    b1 = model.coef_
    return b0, b1

def main():
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    b_exact = linear_regression_exact(x, y)
    b_gd = linear_regression_gd(x,y)
    b_scikit = linear_regression_scikit(x,y)

    print(b_exact)
    print(b_gd)
    print(b_scikit)
    plots(x,y,b_exact)
    plots(x,y,b_gd)
    plots(x,y,b_scikit) #match the exact ~100%

    plt.show()
    
if __name__ == '__main__':
    main()