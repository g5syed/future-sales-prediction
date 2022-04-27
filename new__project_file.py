#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 11:33:15 2020

@author: syed
"""
# importing tools
import numpy as np
import pandas as pd
import datetime as dt
import plotly 
import plotly.offline
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor

# importing data file
data = pd.read_csv("/home/syed/ML-Notebook/Freelance-Projects/FL-P3--date-vs-predicted-sale/df_with_leads.csv", skiprows=0, parse_dates=['date'])
date = data['date']
# dropping `lead` columns
data = data.drop(['market_price_usd_lead_1',
                          'mining_difficulty_lead_1',
                          'hash_rate_lead_1',
                          'blockchain_txns_lead_1',
                          'unique_addresses_lead_1'], axis=1)
      
# Converting `date` datatype from datetime64 to numerical values
data['date'] = pd.to_datetime(data['date'])
data['date']=data['date'].map(dt.datetime.toordinal)

# Splitting in X and y
X = data.drop('diff_market_price_percent',axis=1)
y = data['diff_market_price_percent']

# Spliting into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, shuffle=False)
X_train_date, X_test_date= train_test_split(date, test_size =0.2, shuffle=False)



###Hyerparameters tuning with RandomizedSearchCV
## Different RandomForestRegressor hyperparameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# Instantiate RandomizedSearchCV model
# create random forest classifier model
#rf_model = RandomForestRegressor()

# set up random search meta-estimator
# this will train 100 models over 5 folds of cross validation (500 models total)
#clf = RandomizedSearchCV(rf_model, random_grid, n_iter=100, cv=3, random_state=1)

# train the random search meta-estimator to find the best model out of 100 candidates
#model = clf.fit(X, y)

# print winning set of hyperparameters
#from pprint import pprint
#pprint(model.best_estimator_.get_params())


# Creating function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    scores = {"Training Mean Absolute Error": mean_absolute_error(y_train, train_preds),
              "Test Mean Absolute Error": mean_absolute_error(y_test, test_preds),
              "Training R^2 score": r2_score(y_train, train_preds),
              "Test data R^2 score": r2_score(y_test, test_preds)}
    return scores

# Evaluate the RandomizedSearch model
#show_scores(model)


## RandomizedSearchCV suggests using following parameters
best_model = RandomForestRegressor(bootstrap= True,
                                   ccp_alpha= 0.0,
                                   criterion= 'mse',
                                   max_depth= 10,
                                   max_features= 'auto',
                                   max_leaf_nodes= None,
                                   max_samples= None,
                                   min_impurity_decrease= 0.0,
                                   min_impurity_split= None,
                                   min_samples_leaf= 2,
                                   min_samples_split= 5,
                                   min_weight_fraction_leaf= 0.0,
                                   n_estimators= 900,
                                   n_jobs= None,
                                   oob_score= False,
                                   random_state= None,
                                   verbose= 0,
                                   warm_start= False)

## Fitting and evaluating our new model
best_model.fit(X_train, y_train)
print(show_scores(best_model))

y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)
print(type(y_pred_train))
y_pred = np.concatenate((y_pred_train, y_pred_test))
####Plotting date vs Y with offline plotly


fig = go.Figure()
fig.add_trace(go.Scatter(x=X_test_date,
                         y=y_pred,
                         mode='lines',
                         name='Predicted Labels'))

fig.add_trace(go.Scatter(x=X_train_date, y=y_pred_train,
                         mode='lines',
                         name='Trained Lables'))


fig.update_layout(
    title="DATE vs PREDICTED PERCENT",
    xaxis_title="DATE",
    yaxis_title="Market-Price-Percent",
    legend_title="Legend")

plotly.offline.plot(fig, filename='figure.html')
