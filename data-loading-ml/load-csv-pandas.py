from pandas import read_csv

path = r"../dataset/Iris.csv"
data = read_csv(path)
print(data.shape)
print(data[:3])

path = r"../dataset/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
print(data.shape)
print(data[:3])
