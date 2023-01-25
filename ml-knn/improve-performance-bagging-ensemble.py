from pandas import read_csv
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

path = r'../dataset/pima-indians-diabetes.csv'
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:, 0:8]
y = array[:, 8]
print("{} {}".format(X.shape, y.shape))

seed = 7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
cart = DecisionTreeClassifier()
num_trees = 150
# for bagging classifier
# model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

# for random forest
# model = RandomForestClassifier(n_estimators=num_trees, max_features=5)

# for extra tree
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=5)
result = cross_val_score(model, X, y, cv=kfold)
print(result.mean())
