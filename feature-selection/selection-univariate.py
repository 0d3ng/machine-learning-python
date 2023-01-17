from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, chi2

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(path, names=names)
array = df.values

X = array[:, 0:8]
Y = array[:, 8]

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
set_printoptions(precision=2)
print(fit.scores_)
featured_data = fit.transform(X)
print("Featured data: {}".format(featured_data[0:4]))
