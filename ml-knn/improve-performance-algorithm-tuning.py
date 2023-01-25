from pandas import read_csv
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import uniform

path = r'../dataset/pima-indians-diabetes.csv'
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:, 0:8]
y = array[:, 8]
print("{} {}".format(X.shape, y.shape))

# alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
# print(alphas)
# param_grid = dict(alphas)

model = Ridge()
grid = GridSearchCV(estimator=model, param_grid={'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0]})
grid.fit(X, y)
print("{} {}".format(grid.best_score_, grid.best_estimator_.alpha))

param_grid = {'alpha': uniform()}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, random_state=7)
random_search.fit(X, y)
print("{} {}".format(random_search.best_score_, random_search.best_estimator_.alpha))
