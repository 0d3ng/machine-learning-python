import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from six import StringIO
from IPython.display import Image
import pydotplus

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(r'../dataset/pima-indians-diabetes.csv', header=None, names=col_names)
pima.head()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion matrix {}".format(result))
result1 = classification_report(y_test, y_pred)
print("Classification report {}".format(result1))
result2 = accuracy_score(y_test, y_pred)
print("Accuracy score {}".format(result2))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols,
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('pima_diabetes_tree.png')
Image(graph.create_png())
