from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve

# for the plot:
import seaborn as sns
import matplotlib.pyplot as plt

ds = load_iris() #Load and return the iris dataset (classification).


#Roc curves are used only in BINARY classification, in this case we know that the classes are 3, for this reason we exclude the first 50 elements
X = ds['data'][50:150]
y = ds['target'][50:150]

X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.3) #normally a 30/70 or 20/80 split is used, returns

model = LogisticRegression() #plain vanilla Logistic Regression
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
pred_train = model.predict(X_train)

plot_roc_curve(model, X_test, y_test)
plt.show()
plt.savefig('stuff/Roc_Curve.png')