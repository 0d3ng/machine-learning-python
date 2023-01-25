from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, log_loss

X_actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
y_predict = [1, 0, 1, 1, 1, 0, 1, 1, 0, 0]

print("confusion matrix     : ", confusion_matrix(X_actual, y_predict))
print("Accuracy scores      : ", accuracy_score(X_actual, y_predict))
print("Classification report: ", classification_report(X_actual, y_predict))
print("Roc auc score        : ", roc_auc_score(X_actual, y_predict))
print("Log loss             : ", log_loss(X_actual, y_predict))
