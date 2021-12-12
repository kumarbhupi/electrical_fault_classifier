import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_csv('../data/detect_dataset.csv')
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
