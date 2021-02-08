from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import os

ds = load_iris()


# we are going to load all the dataset and take only 2 of the 4 possible features:
X=ds['data']
X = X[: , [0,2]]
y=ds['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)



model = LogisticRegression()

model.fit(X_train, y_train)

p_test = model.predict(X_test)
p_train = model.predict(X_train)


plot_decision_regions(X, y, clf = model)
plt.xlabel('sepal lenght')
plt.ylabel('sepal width')
plt.savefig('stuff/model_decision_regions')
# the figure is saved in the stuff repository
plt.show()
