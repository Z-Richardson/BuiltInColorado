#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:46:28 2020

@title: Plots_and_Predictions
@author: Zachary Richardson, Ph.D.
@notes: Dimensionality Reduction Plots + Predicting Job Subcategory
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
os.chdir('./BuiltInColorado/')
# =============================================================================
# Load All Data
# =============================================================================
# Main Data
with open('python_ml/pickles/df.pickle', 'rb') as data:
    df = pickle.load(data)

# X_train
with open('python_ml/pickles/X_train.pickle', 'rb') as data:
    X_train = pickle.load(data)

# X_test
with open('python_ml/pickles/X_test.pickle', 'rb') as data:
    X_test = pickle.load(data)

# y_train
with open('python_ml/pickles/y_train.pickle', 'rb') as data:
    y_train = pickle.load(data)

# y_test
with open('python_ml/pickles/y_test.pickle', 'rb') as data:
    y_test = pickle.load(data)
    
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
# Concatenate Train and Test Data   
# =============================================================================
features = np.concatenate((features_train,features_test), axis=0)
labels = np.concatenate((labels_train,labels_test), axis=0)
    

# =============================================================================
# Create PCA Plots   
# =============================================================================   
def plot_dim_red(model, features, labels, n_components=2):
    # Creation of the model
    if (model == 'PCA'):
        mod = PCA(n_components=n_components)
        title = "PCA decomposition"  # for the plot
        
    elif (model == 'TSNE'):
        mod = TSNE(n_components=2)
        title = "t-SNE decomposition" 

    else:
        return "Error"
    
    # Fit and transform the features
    principal_components = mod.fit_transform(features)
    
    # Put them into a dataframe
    df_features = pd.DataFrame(data=principal_components,
                     columns=['PC1', 'PC2'])
    df_labels = pd.DataFrame(data=labels,
                             columns=['label'])
    
    df_full = pd.concat([df_features, df_labels], axis=1)
    # df_full['label'] = df_full['label'].astype(str)

    # Get labels name
    category_names = {
        0: 'Analysis & Reporting',
        1: 'Analytics',
        2: 'Data Engineering',
        3: 'Data Science'
    }

    # # And map labels
    df_full['label_name'] = df_full['label']
    df_full = df_full.replace({"label_name": category_names}) 


    # Plot
    plt.figure(figsize=(10,10))
    sns.scatterplot(x='PC1',
                    y='PC2',
                    hue="label_name", 
                    data=df_full,
                    palette=sns.color_palette("bright", 4),
                    alpha=.7).set_title(title)
    
    
plot_dim_red("PCA", features=features, labels=labels, n_components=2)
plot_dim_red("TSNE", features=features, labels=labels, n_components=2)
    
# =============================================================================
# Test Gradient Boosting Model
# =============================================================================    
with open('python_ml/models/best_gbc.pickle', 'rb') as data:
    gbc_model = pickle.load(data)
 
with open('python_ml/models/best_rfc.pickle', 'rb') as data:
   lrc_model = pickle.load(data)    
    
# Category mapping dictionary
category_codes = {
    'Analysis & Reporting': 0,
    'Analytics': 1,
    'Data Engineering': 2,
    'Data Science': 3
}

category_names = {
   0: 'Analysis & Reporting',
   1: 'Analytics',
   2: 'Data Engineering',
   3: 'Data Science'
}

    
gbc_predictions = gbc_model.predict(features_test)
lrc_predictions = lrc_model.predict(features_test)    

# Indexes of the test set
index_X_test = X_test.index

# We get them from the original df
df_test = df.loc[index_X_test]

# Add the predictions
df_test['GBC_Pred'] = gbc_predictions
df_test['LRC_Pred'] = lrc_predictions

# Clean columns
df_test = df_test[['Job_Title', 'Description', 'Job_Subcategory', 
                   'Job_Subcategory_Code', 'GBC_Pred', 'LRC_Pred']]

# Decode
df_test['GBC_Subcategory_Pred'] = df_test['GBC_Pred']
df_test = df_test.replace({'GBC_Subcategory_Pred':category_names})
df_test['LRC_Subcategory_Pred'] = df_test['LRC_Pred']
df_test = df_test.replace({'LRC_Subcategory_Pred':category_names})

# Clean columns again
df_test = df_test[['Job_Title', 'Description', 'Job_Subcategory', 
                   'GBC_Subcategory_Pred', 'LRC_Subcategory_Pred']]
    

cond1 = (df_test['Job_Subcategory'] != df_test['GBC_Subcategory_Pred'])
cond2 = (df_test['Job_Subcategory'] != df_test['LRC_Subcategory_Pred'])

condition = pd.DataFrame(dict(cond1 = cond1, cond2 = cond2))

df_misclassified1 = df_test[cond1]
df_misclassified2 = df_test[cond2]


df_misclassified = pd.concat([df_misclassified1, df_misclassified2])
df_misclassified = df_misclassified.drop_duplicates(keep='first')


    
