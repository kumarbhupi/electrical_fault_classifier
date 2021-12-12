from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from joblib import dump

clf_svc_poly = SVC(probability=True, kernel='poly', degree=2)
clf_svc_poly.fit(X_train, y_train)
dump(clf_svc_poly, '../models/clf_svc_poly.joblib')


clf_svc_rbf = SVC(probability=True, kernel = 'rbf')
clf_svc_rbf.fit(X_train, y_train)
dump(clf_svc_rbf, '../models/clf_svc_rbf.joblib')


clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
dump(clf_lr, '../models/clf_lr.joblib')


clf_gbc = GradientBoostingClassifier()
clf_gbc.fit(X_train, y_train)
dump(clf_gbc, '../models/clf_gbc.joblib')


clf_dtc = DecisionTreeClassifier()
clf_dtc.fit(X_train, y_train)
dump(clf_dtc, '../models/clf_dtc.joblib')

clf_mlp = MLPClassifier()
clf_mlp.fit(X_train, y_train)
dump(clf_mlp, '../models/clf_mlp.joblib')
