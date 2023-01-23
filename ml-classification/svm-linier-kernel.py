from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.50)
model = SVC(kernel='linier', C=1E10)
model.fit(X, y)

# model = SVC(C=10000000000, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
#             gamma='auto_deprecated', kernel='linier', max_iter=-1, probability=False, random_state=None, shrinking=True,
#             tol=0.001, verbose=False)


def decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
