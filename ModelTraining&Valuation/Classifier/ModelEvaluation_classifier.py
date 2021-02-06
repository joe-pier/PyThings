# to evaluate a classifier, as we said in the post we can use different tecniques: the first an most important is the confusion matrix
# when the confusion matrix is done we can calculate some indices and make some cool stuff

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


# lets do our confusion matrix:
confusionmatrix= confusion_matrix(y_test, pred_test)
sns.heatmap(confusionmatrix, annot=True)
#plt.show()
plt.savefig('stuff/confusionmatrix.png')

# lets calculate all the important indices:
print(classification_report(y_test, pred_test))
# you should see some sort of table: you can see the precision, recall, f1-score & support




