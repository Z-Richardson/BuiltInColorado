#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:46:40 2020

@title: Random_Forest_Modeling
@author: Zachary Richardson, Ph.D.
@notes: Examine the Random Forest model
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
os.chdir('/Users/Admin/Dropbox/Post-Doc Files/The Lab (Blog Site)/Job Post Keyword Searches/')
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
# ---------------------------------  
# Check data to be sure everything looks as expected
print(features_train.shape)
print(features_test.shape)       
# =============================================================================
# Cross-Validation for Hyperparameter Tuning    
# =============================================================================
rf_0 = RandomForestClassifier(random_state = 42)

# Tune select parameters for speed and based on sample size
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Combine features to create the grid to use in estimation
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)
    
# ---------------------------------  
# Randomized Search Cross Validation
# First create the base model to tune
rfc = RandomForestClassifier(random_state = 42)

# Definition of the random search
random_search = RandomizedSearchCV(estimator = rfc,
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
# Based on results now re-tune grid to the suggested parameters
bootstrap = [False]
max_depth = [30, 40, 50]
max_features = ['auto']
min_samples_leaf = [1, 2, 4]
min_samples_split = [5, 10, 15]
n_estimators = [400]

param_grid = {
    'bootstrap': bootstrap,
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators
}    
    
# Fit model now based on new grid:
rfc = RandomForestClassifier(random_state = 42)

# Manually create the splits in CV in order to be able to fix a random_state
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rfc, 
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = cv_sets,
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)
print("---------------------------------")   
print("")  
print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)
# --------------------------------- 
# Save results as best fit for now:
best_rfc = grid_search.best_estimator_
# =============================================================================
# Fit Model and Compare Performance    
# =============================================================================
# Check Best Fit Model Info
best_rfc.fit(features_train, labels_train)
    
# Use Best Fit Model to Make Predictions
rfc_pred = best_rfc.predict(features_test)
# --------------------------------- 
# Now check accuracy of predictions on training and test data:
print("---------------------------------")   
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_rfc.predict(features_train)))
print("---------------------------------")    
print("The test accuracy is: ")
print(accuracy_score(labels_test, rfc_pred))    
    
# # Classification report
# print("Classification report")
# print(classification_report(labels_test,rfc_pred))       
    
# # Confusion matrix
# aux_df = df[['Job_Subcategory', 'Job_Subcategory_Code']].drop_duplicates().sort_values('Job_Subcategory_Code')
# conf_matrix = confusion_matrix(labels_test, rfc_pred)
# # plt.figure(figsize=(12.8,6))
# sns.heatmap(conf_matrix, 
#             annot=True,
#             xticklabels=aux_df['Job_Subcategory'].values, 
#             yticklabels=aux_df['Job_Subcategory'].values,
#             cmap="Blues")
# plt.ylabel('Predicted')
# plt.xlabel('Actual')
# plt.title('Confusion matrix')
# plt.ylim(-0.5, 4.5)
# plt.show()    

# =============================================================================
# Save Model Results to Use Later
# =============================================================================
# Create Dataframe for Model Results
d = {
     'Model': 'Random Forest',
     'Training Set Accuracy': accuracy_score(labels_train, best_rfc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, rfc_pred)
}

df_models_rfc = pd.DataFrame(d, index=[0])  
  
# Save Model Results for Analysis Later
with open('python_ml/models/best_rfc.pickle', 'wb') as output:
    pickle.dump(best_rfc, output)
    
with open('python_ml/models/df_models_rfc.pickle', 'wb') as output:
    pickle.dump(df_models_rfc, output)
# =============================================================================
# =============================================================================
# =============================================================================