from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# for the plot:
import seaborn as sns
import matplotlib.pyplot as plt

ds = load_iris() #Load and return the iris dataset (classification).

X = ds['data']
y = ds['target']

X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.3) #normally a 30/70 or 20/80 split is used, returns

model = LogisticRegression() #plain vanilla Logistic Regression
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
pred_train = model.predict(X_train)