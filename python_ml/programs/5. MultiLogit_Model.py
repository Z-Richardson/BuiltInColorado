#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:14:08 2020

@title: MultiLogit_Model
@author: Zachary Richardson, Ph.D.
@notes: Multinomial Logistic Regression Models
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
lr_0 = LogisticRegression(random_state = 42)

print('Parameters currently in use:\n')
pprint(lr_0.get_params())
# ---------------------------------  
C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)]
multi_class = ['multinomial']
solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
class_weight = ['balanced', None]
penalty = ['l2']
random_grid = {'C': C,
               'multi_class': multi_class,
               'solver': solver,
               'class_weight': class_weight,
               'penalty': penalty}

pprint(random_grid)
# ---------------------------------  
# Randomized Search Cross Validation
# First create the base model to tune
lrc = LogisticRegression(random_state = 42)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=lrc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)

# Fit the random search model
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
C = [float(x) for x in np.linspace(start = 0.9, stop = 1, num = 10)]
multi_class = ['multinomial']
solver = ['lbfgs']
class_weight = [None]
penalty = ['l2']

param_grid = {'C': C,
               'multi_class': multi_class,
               'solver': solver,
               'class_weight': class_weight,
               'penalty': penalty}

# Create a base model
lrc = LogisticRegression(random_state = 42)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=lrc, 
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
best_lrc = grid_search.best_estimator_
# =============================================================================
# Fit Model and Compare Performance    
# =============================================================================
# Check Best Fit Model Info
best_lrc.fit(features_train, labels_train)
    
# Use Best Fit Model to Make Predictions
lrc_pred = best_lrc.predict(features_test)
# --------------------------------- 
# Now check accuracy of predictions on training and test data:
print("---------------------------------")   
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_lrc.predict(features_train)))
print("---------------------------------")    
print("The test accuracy is: ")
print(accuracy_score(labels_test, lrc_pred))
# =============================================================================
# Save Model Results to Use Later
# =============================================================================
# Create Dataframe for Model Results
d = {
     'Model': 'Logistic Regression',
     'Training Set Accuracy': accuracy_score(labels_train, best_lrc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, lrc_pred)
}

df_models_lrc = pd.DataFrame(d, index=[0])
# --------------------------------- 
# Save Model Results for Analysis Later   
with open('python_ml/models/best_lrc.pickle', 'wb') as output:
    pickle.dump(best_lrc, output)
    
with open('python_ml/models/df_models_lrc.pickle', 'wb') as output:
    pickle.dump(df_models_lrc, output)
# =============================================================================
# =============================================================================
# =============================================================================
