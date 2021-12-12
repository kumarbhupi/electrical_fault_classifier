from joblib import load

clf_svc_poly = load('clf_svc_poly.joblib')
print('SCV POLY Score:', clf_svc_poly.score(X_test, y_test))


clf_svc_rbf = load('clf_svc_rbf.joblib')
print('SCV RBF Score:', clf_svc_rbf.score(X_test, y_test))


clf_lr = load('clf_lr.joblib')
print('Logistic Regression Score:', clf_lr.score(X_test, y_test))


clf_gbc = load('clf_gbc.joblib')
print('Gradient Boosting Classifier Score:', clf_gbc.score(X_test, y_test))


clf_dtc = load('clf_dtc.joblib')
print('Decision Tree Classifier Score:', clf_dtc.score(X_test, y_test))


clf_mlp = load('clf_mlp.joblib')
print('Perceptron Score:', clf_mlp.score(X_test, y_test))
