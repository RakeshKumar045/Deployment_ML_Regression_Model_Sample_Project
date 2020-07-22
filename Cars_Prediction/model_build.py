#!/usr/bin/env python
# coding: utf-8

import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

car = pd.read_csv('datasets_33080_43333_car data.csv')

car['Current_Year'] = 2020
car['No_of_years'] = car['Current_Year'] - car['Year']

cars = car[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner',
            'Current_Year', 'No_of_years']]

cars.drop(['Year'], axis=1, inplace=True)
cars.drop(['Current_Year'], axis=1, inplace=True)
cars = pd.get_dummies(cars, drop_first=True)

X = cars.iloc[:, 1:]  ## from first index everything is independent feature
y = cars.iloc[:, 0]

model = ExtraTreesRegressor()
model.fit(X, y)

print(model.feature_importances_)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Random forest
rf = RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)

# Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

## always give as key: value

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

## apply RandomsearchCV

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               scoring='neg_mean_squared_error',
                               n_iter=10,
                               cv=5,
                               verbose=2,  ## to display results
                               random_state=42,
                               n_jobs=1)

## Now we can use the result of RandomSearchCV to fit X and y

rf_random.fit(X_train, y_train)

predictions = rf_random.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

## model is created. Now,pickle this file and put it as a pickle file in write byte mode

# open a file, where you ant to store the data
file = open('car_predict_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
