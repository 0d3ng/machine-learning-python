import csv
import numpy as np

path = r"../dataset/Iris.csv"
with open(path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = list(reader)
    data = np.array(data)
    print(headers)
    print(data.shape)
    print(data[:3])
