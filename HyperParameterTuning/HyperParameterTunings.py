from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import pandas as pd
from operator import itemgetter

ds = load_boston()

X = ds['data']
y = ds['target']


model = RandomForestRegressor() #lets keep the model simple to make everyone understand, if you want you can change it with whatever you want

print(model.get_params()) #to print all the hyperparameters available


# lets do some grid search hyperparameter tuning:


# lets choose different values for two random picked parameters (try to use different ones):
n_estimators = [10, 50, 100]
max_depth = [3, 10, 20]

grid = {'n_estimators':n_estimators, 'max_depth': max_depth}

gs = GridSearchCV(estimator = model, param_grid=grid, cv = 4) #default cross validation used is a 5 fold cross validation

gs.fit(X,y)




results = gs.cv_results_

results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by = ['rank_test_score'])
results_df_sorted.to_csv('results.csv')