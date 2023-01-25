from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion

path = r'../dataset/pima-indians-diabetes.csv'
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
X = data.iloc[:, :-1].values
y = data.iloc[:, 8].values
print("{} {}".format(X.shape, y.shape))

features = [('pca', PCA(n_components=3)), ('select_best', SelectKBest(k=6))]
feature_union = FeatureUnion(features)

estimators = [('feature_union', feature_union), ('logistic', LogisticRegression())]
model = Pipeline(estimators)

kFold = KFold(n_splits=20, random_state=7, shuffle=True)
result = cross_val_score(model, X, y, cv=kFold)
print("{0:.2f}".format(result.mean()))
