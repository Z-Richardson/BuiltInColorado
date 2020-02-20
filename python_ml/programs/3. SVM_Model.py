#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:46:28 2020

@title: SVM_Model
@author: Zachary Richardson, Ph.D.
@notes: Support Vector Machine Model
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
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
svc_0 = svm.SVC(random_state = 42)

# Tune select parameters for speed and based on sample size
C = [.0001, .001, .01]
gamma = [.0001, .001, .01, .1, 1, 10, 100]
degree = [1, 2, 3, 4, 5]
kernel = ['linear', 'rbf', 'poly']
probability = [True]

# Create the random grid
random_grid = {'C': C,
              'kernel': kernel,
              'gamma': gamma,
              'degree': degree,
              'probability': probability
             }

pprint(random_grid)
# ---------------------------------  
# Randomized Search Cross Validation
# First create the base model to tune
svc = svm.SVC(random_state = 42)

# Definition of the random search
random_search = RandomizedSearchCV(estimator = svc,
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
C = [.0001, .001, .01, .1]
degree = [2, 3, 4]
gamma = [1, 10, 100]
probability = [True]

param_grid = [
  {'C': C, 'kernel':['linear'], 'probability':probability},
  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
]

# Create a base model
svc = svm.SVC(random_state = 42)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, 
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

# --------------------------------- 
# Random = Best for SVC
# Save Best SVC Model:
best_svc = random_search.best_estimator_
# =============================================================================
# Fit Model and Compare Performance    
# =============================================================================
# Check Best Fit Model Info
best_svc.fit(features_train, labels_train)
    
# Use Best Fit Model to Make Predictions
svc_pred = best_svc.predict(features_test)
# --------------------------------- 
# Now check accuracy of predictions on training and test data:
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_svc.predict(features_train)))
print("---------------------------------")    
print("The test accuracy is: ")
print(accuracy_score(labels_test, svc_pred))  
# =============================================================================
# Save Model Results to Use Later
# =============================================================================
# Create Dataframe for Model Results
d = {
     'Model': 'SVM',
     'Training Set Accuracy': accuracy_score(labels_train, best_svc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, svc_pred)
}

df_models_svc = pd.DataFrame(d, index=[0])
# --------------------------------- 
# Save Model Results for Analysis Later   
with open('python_ml/models/best_svc.pickle', 'wb') as output:
    pickle.dump(best_svc, output)
    
with open('python_ml/models/df_models_svc.pickle', 'wb') as output:
    pickle.dump(df_models_svc, output)
# =============================================================================
# =============================================================================
# =============================================================================
