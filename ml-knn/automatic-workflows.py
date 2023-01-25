from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

path = r'../dataset/pima-indians-diabetes.csv'
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
X = data.iloc[:, :-1].values
y = data.iloc[:, 8].values
print("{} {}".format(X.shape, y.shape))
estimators = [('standardize', StandardScaler()), ('lda', LinearDiscriminantAnalysis())]
model = Pipeline(estimators)

kFold = KFold(n_splits=20, random_state=7, shuffle=True)
result = cross_val_score(model, X, y, cv=kFold)
print("{0:.2f}".format(result.mean()))
