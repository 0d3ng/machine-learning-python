import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, names=headernames)
dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Confussion matrix {}".format(confusion_matrix(y_test, y_pred)))
print("Classification report {}".format(classification_report(y_test, y_pred)))
print("Accuracy {}".format(accuracy_score(y_test, y_pred)))
