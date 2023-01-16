from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(path, names=names)
array = df.values
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(array)
set_printoptions(precision=1)
print(data_scaled[0:10])
