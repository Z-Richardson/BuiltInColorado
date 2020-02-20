#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:39:52 2020

@title: Model Compare
@author: Zachary Richardson, Ph.D.
@notes: Compare model accuracy between different estimated models
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import pandas as pd
import os
os.chdir('./BuiltInColorado/')
# =============================================================================
# Set-Up Data
# =============================================================================
path_pickles = "./BuiltInColorado/python_ml/models/"

list_pickles = [
    "df_models_gbc.pickle",
    "df_models_lrc.pickle",
    "df_models_mnbc.pickle",
    "df_models_rfc.pickle",
    "df_models_svc.pickle"
]

df_summary = pd.DataFrame()
# ---------------------------------  
for pickle_ in list_pickles:
    path = path_pickles + pickle_
    with open(path, 'rb') as data:
        df = pickle.load(data)
    df_summary = df_summary.append(df)

df_summary = df_summary.reset_index().drop('index', axis=1)
# ---------------------------------  
df_summary
# ---------------------------------  
df_summary.sort_values('Test Set Accuracy', ascending=False)
# =============================================================================
# =============================================================================
# =============================================================================
