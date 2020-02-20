#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:14:08 2020

@title: MultinomialNB_Model
@author: Zachary Richardson, Ph.D.
@notes: Multinomial Naïve Bayes Models
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
# =============================================================================
# Cross-Validation for Hyperparameter Tuning    
# =============================================================================
mnbc = MultinomialNB()
mnbc
# ---------------------------------  
mnbc.fit(features_train, labels_train)
mnbc_pred = mnbc.predict(features_test)

print("---------------------------------")
print("The training accuracy is: ")
print(accuracy_score(labels_train, mnbc.predict(features_train)))
print("---------------------------------")
print("The test accuracy is: ")
print(accuracy_score(labels_test, mnbc_pred))
# =============================================================================
# Save Model Results to Use Later
# =============================================================================
# Create Dataframe for Model Results
d = {
     'Model': 'Multinomial Naïve Bayes',
     'Training Set Accuracy': accuracy_score(labels_train, mnbc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, mnbc_pred)
}

df_models_mnbc = pd.DataFrame(d, index=[0])
# --------------------------------- 
# Save Model Results for Analysis Later   
with open('python_ml/models/best_mnbc.pickle', 'wb') as output:
    pickle.dump(mnbc, output)
    
with open('python_ml/models/df_models_mnbc.pickle', 'wb') as output:
    pickle.dump(df_models_mnbc, output)
# =============================================================================
# =============================================================================
# =============================================================================