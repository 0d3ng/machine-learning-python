from pandas import read_csv
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

path = r'../dataset/pima-indians-diabetes.csv'
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:, 0:8]
y = array[:, 8]
print("{} {}".format(X.shape, y.shape))

seed = 7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

estimators = [('logistic', LogisticRegression()), ('cart', DecisionTreeClassifier()), ('svm', SVC())]

model = VotingClassifier(estimators)

result = cross_val_score(model, X, y, cv=kfold)
print(result.mean())
