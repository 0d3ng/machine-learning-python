from pandas import read_csv
from pandas import set_option

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
set_option('display.width', 100)
set_option('display.precision', 2)
print(data.shape)
print(data.describe())

# review class distribution
count_class = data.groupby('class').size()
print(count_class)

# review correlation between attribute
correlations = data.corr(method='pearson')
print(correlations)

# review skew of attribute distribution
print(data.skew())
