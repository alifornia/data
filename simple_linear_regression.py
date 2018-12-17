import sys
import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    n = np.size(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    SS_xy = np.sum(x*y) - n*x_mean*y_mean
    SS_xx = np.sum(x*x) - n*x_mean*x_mean
    
    b1 = SS_xy/SS_xx
    b0 = y_mean - b1*x_mean
    return b0, b1

def plots(x, y, b):
    
    plt.scatter(x,y)
    plt.plot(x,b[0]+b[1]*x)
    plt.show()


def main():
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    
    b = estimate_coef(x, y)
    
    print(b)
    plots(x,y,b)
    
if __name__ == '__main__':
    
    main()