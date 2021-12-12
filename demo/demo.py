import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/detect_dataset.csv')
dataset = dataset.drop(['Unnamed: 7', 'Unnamed: 8'], axis=1)

Y = dataset.values[:, 0]
X = dataset.values[:, 1:]


def reduce_dimensions(X, n_dim):
    pca = PCA(n_components=n_dim)
    return pca.fit_transform(X)


X_reduced_default = reduce_dimensions(X, 2)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X_reduced_default[:, 0], X_reduced_default[:, 1], c=Y, cmap=plt.cm.coolwarm, s=.6, alpha=0.3)

X_train, X_test, y_train, y_test = train_test_split(X_reduced_default, Y, test_size=0.33, random_state=42)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from joblib import dump

clf_svc_poly = SVC(probability=True, kernel='poly', degree=2)
clf_svc_poly.fit(X_train, y_train)
dump(clf_svc_poly, 'models/clf_svc_poly.joblib')


clf_svc_rbf = SVC(probability=True, kernel = 'rbf')
clf_svc_rbf.fit(X_train, y_train)
dump(clf_svc_rbf, 'models/clf_svc_rbf.joblib')


clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
dump(clf_lr, 'models/clf_lr.joblib')


clf_gbc = GradientBoostingClassifier()
clf_gbc.fit(X_train, y_train)
dump(clf_gbc, 'models/clf_gbc.joblib')


clf_dtc = DecisionTreeClassifier()
clf_dtc.fit(X_train, y_train)
dump(clf_dtc, 'models/clf_dtc.joblib')

clf_mlp = MLPClassifier()
clf_mlp.fit(X_train, y_train)
dump(clf_mlp, 'models/clf_mlp.joblib')


from joblib import load

clf_svc_poly = load('models/clf_svc_poly.joblib')
print('SCV POLY Score:', clf_svc_poly.score(X_test, y_test))


clf_svc_rbf = load('models/clf_svc_rbf.joblib')
print('SCV RBF Score:', clf_svc_rbf.score(X_test, y_test))


clf_lr = load('models/clf_lr.joblib')
print('Logistic Regression Score:', clf_lr.score(X_test, y_test))


clf_gbc = load('models/clf_gbc.joblib')
print('Gradient Boosting Classifier Score:', clf_gbc.score(X_test, y_test))


clf_dtc = load('models/clf_dtc.joblib')
print('Decision Tree Classifier Score:', clf_dtc.score(X_test, y_test))


clf_mlp = load('models/clf_mlp.joblib')
print('Perceptron Score:', clf_mlp.score(X_test, y_test))
