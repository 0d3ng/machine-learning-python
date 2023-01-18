from pandas import read_csv
from sklearn.decomposition import PCA

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(path, names=names)
array = df.values
X = array[:, 0:8]
Y = array[:, 8]

pca = PCA(n_components=3)
fit = pca.fit(X)
print("Explained variance: {}".format(fit.explained_variance_ratio_))
print(fit.components_)
