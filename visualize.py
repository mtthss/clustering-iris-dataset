import matplotlib.pyplot as plt
import numpy as np

__author__ = 'matteo'

# scatter plots for the projection on each pair of factors
def pairwise_scatter(X,y):

    m = X.shape[1]
    f, axarr = plt.subplots(m,m)

    for i in xrange(X.shape[1]):
        for j in xrange(X.shape[1]):
            axarr[i, j].scatter(X[0:49, i], X[0:49, j], c="r")
            axarr[i, j].scatter(X[50:99, i], X[50:99, j], c="g")
            axarr[i, j].scatter(X[100:149, i], X[100:149, j], c="b")
            axarr[i, j].set_title('factors:'+str(i+1)+' vs '+str(i+1))

    plt.show()
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)


# plot transformed 2D representations
def plot_2D_transformation(X):

    plt.figure()
    plt.scatter(X[0:49, 0], X[0:49, 1], c="r")
    plt.scatter(X[50:99, 0], X[50:99, 1], c="g")
    plt.scatter(X[100:149, 0], X[100:149, 1], c="b")
    plt.gray()
    plt.show()

def plot_2D_clustering(X, labels):

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels.astype(np.float))
    plt.gray()
    plt.show()
