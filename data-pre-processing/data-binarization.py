from pandas import read_csv
from sklearn.preprocessing import Binarizer,StandardScaler
from numpy import set_printoptions

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(path, names=names)
array = df.values

binarizer = Binarizer(threshold=0.5).fit(array)
binarized = binarizer.transform(array)
print(binarized[0:5])

# standardization
data_scaler = StandardScaler().fit(array)
data_rescaled = data_scaler.transform(array)
set_printoptions(precision=2)
print(data_rescaled[0:5])