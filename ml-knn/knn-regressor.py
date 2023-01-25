import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(path, names=headernames)
print(dataset.head())
array = dataset.values

X = array[:, :2]
y = array[:, 2]
print(dataset.shape)

classifier = KNeighborsRegressor(n_neighbors=8)
classifier.fit(X, y)
print("The MSE is:", format(np.power(y - classifier.predict(X), 2).mean()))