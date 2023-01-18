from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(path, names=names)
array = df.values

X = array[:, 0:8]
Y = array[:, 8]

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(X, Y)
print("Number of feature: {}".format(rfe.n_features_))
print("Selected feature : {}".format(rfe.n_features_in_))
print("Feature ranking  : {}".format(rfe.ranking_))
