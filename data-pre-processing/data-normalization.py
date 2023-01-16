from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(path, names=names)
array = df.values

data_normalizer = Normalizer(norm="l1").fit(array)
data_normalized = data_normalizer.transform(array)
set_printoptions(precision=2)
print(data_normalized[0:3])
