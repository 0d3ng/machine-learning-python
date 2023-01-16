from pandas import read_csv

path = r"../dataset/Iris.csv"
data = read_csv(path)
print(data.shape)
print(data.dtypes)
