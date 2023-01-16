from matplotlib import pyplot as plt
from pandas import read_csv

path = r"../dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
data.hist()
plt.show()

# density plots
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()

# box plot
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.show()
