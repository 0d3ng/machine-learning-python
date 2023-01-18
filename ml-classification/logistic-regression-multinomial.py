import sklearn
from sklearn import datasets
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.4)
digreg = linear_model.LogisticRegression()
digreg.fit(X_train, y_train)
y_pred = digreg.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred) * 100)
