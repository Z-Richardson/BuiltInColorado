#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:14:08 2020

@title: GBM_Model
@author: Zachary Richardson, Ph.D.
@notes: Gradient Boosting Machine Models
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
os.chdir('./BuiltInColorado/')
# =============================================================================
# Load All Data
# =============================================================================
# Main Data
with open('python_ml/pickles/df.pickle', 'rb') as data:
    df = pickle.load(data)

# Training Data Features
with open('python_ml/pickles/features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)

# Training Data Labels
with open('python_ml/pickles/labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)

# Test Data Features
with open('python_ml/pickles/features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)

# Test Data Labels
with open('python_ml/pickles/labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)  
# =============================================================================
# Cross-Validation for Hyperparameter Tuning    
# =============================================================================
gb_0 = GradientBoostingClassifier(random_state = 42)

print('Parameters currently in use:\n')
pprint(gb_0.get_params())
# ---------------------------------  
n_estimators = [200, 800]
max_features = ['auto', 'sqrt']
max_depth = [10, 40]
max_depth.append(None)
min_samples_split = [10, 30, 50]
min_samples_leaf = [1, 2, 4]
learning_rate = [.1, .5]
subsample = [.5, 1.]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate,
               'subsample': subsample}
# ---------------------------------  
# Randomized Search Cross Validation
# First create the base model to tune
gbc = GradientBoostingClassifier(random_state = 42)

# Definition of the random search
random_search = RandomizedSearchCV(estimator = gbc,
                                   param_distributions = random_grid,
                                   n_iter = 50,
                                   scoring = 'accuracy',
                                   cv = 3, 
                                   verbose = 1, 
                                   random_state = 42)

# Fit the random search model (Will take a bit of time)
random_search.fit(features_train, labels_train)    

# Hyperparameter Results
print("---------------------------------")   
print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("---------------------------------")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)
# ---------------------------------  
# Perform a Grid Search Now  
# Create the parameter grid based on the results of random search 
max_depth = [5, 10, 15]
max_features = ['sqrt']
min_samples_leaf = [2]
min_samples_split = [10, 30, 50]
n_estimators = [800]
learning_rate = [.1, .5]
subsample = [1.]

param_grid = {
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'subsample': subsample

}

# Create a base model
gbc = GradientBoostingClassifier(random_state = 42)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=gbc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)

print("---------------------------------")
print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("---------------------------------")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)

# Save Best GBC Model:
best_gbc = grid_search.best_estimator_
# =============================================================================
# Fit Model and Compare Performance    
# =============================================================================
# Check Best Fit Model Info
best_gbc.fit(features_train, labels_train)
    
# Use Best Fit Model to Make Predictions
gbc_pred = best_gbc.predict(features_test)
# --------------------------------- 
# Now check accuracy of predictions on training and test data:
print("---------------------------------")
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_gbc.predict(features_train)))
print("---------------------------------")    
print("The test accuracy is: ")
print(accuracy_score(labels_test, gbc_pred)) 
# =============================================================================
# Save Model Results to Use Later
# =============================================================================
# Create Dataframe for Model Results
d = {
     'Model': 'Gradient Boosting',
     'Training Set Accuracy': accuracy_score(labels_train, best_gbc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, gbc_pred)
}

df_models_gbc = pd.DataFrame(d, index=[0])
# --------------------------------- 
# Save Model Results for Analysis Later   
with open('python_ml/models/best_gbc.pickle', 'wb') as output:
    pickle.dump(best_gbc, output)
    
with open('python_ml/models/df_models_gbc.pickle', 'wb') as output:
    pickle.dump(df_models_gbc, output)
# =============================================================================
# =============================================================================
# =============================================================================
