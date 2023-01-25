from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X_actual = [5, -1, 2, 10]
y_predict = [3.5, -0.9, 2, 9.9]

print("R squared: ", r2_score(X_actual, y_predict))
print("MAE      : ", mean_absolute_error(X_actual, y_predict))
print("MSE      : ", mean_squared_error(X_actual, y_predict))