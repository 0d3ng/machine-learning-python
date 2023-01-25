from pandas import read_csv
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

path = r'../dataset/pima-indians-diabetes.csv'
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:, 0:8]
y = array[:, 8]
print("{} {}".format(X.shape, y.shape))

seed = 5
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
num_trees = 50
# for adaboost classifier
# model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

# for gradient boosting
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)

result = cross_val_score(model, X, y, cv=kfold)
print(result.mean())
