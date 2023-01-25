import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks
from scipy.stats import mode
from sklearn.metrics import accuracy_score

sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show()

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
print("{0:.2f}".format(accuracy_score(digits.target, labels)))
