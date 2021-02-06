from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random

dataset = load_iris() #Load and return the iris dataset (classification).

X = dataset['data']
y = dataset['target']

X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.3) #normally a 30/70 or 20/80 split is used, returns

model = LogisticRegression() #plain vanilla Logistic Regression
model.fit(X_train, y_train)


random_iris = int(random.random())
pred_test = model.predict([X_test[random_iris]])

print(f'features: {X_test[random_iris]} \nreal target: {y_test[random_iris]} \npredicted target: {pred_test}')
