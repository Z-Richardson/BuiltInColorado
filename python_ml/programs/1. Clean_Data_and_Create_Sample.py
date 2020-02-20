#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:07:44 2020

@title: Clean_Data_and_Create_Sample
@author: Zachary Richardson, Ph.D.
@notes: Following along with created example from https://github.com/miguelfzafra
        create the analysis sample that we will use for the ML Analysis
"""
# =============================================================================
# Set-Up Workspace
# =============================================================================
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
import os
os.chdir('./BuiltInColorado/')

# =============================================================================
# Load Data
# =============================================================================
df_path = "./BuiltInColorado/"
df_path2 = df_path + 'jobs_data.csv'
df = pd.read_csv(df_path2, sep=',')

with open('jobs_data.pickle', 'wb') as output:
    pickle.dump(df, output)

# Use below if you ever need to reload the data, not done for right now:
# path_df = "/Users/Admin/Dropbox/Post-Doc Files/The Lab (Blog Site)/Job Post Keyword Searches/jobs_data.pickle"
# with open(path_df, 'rb') as data:
#     df = pickle.load(data)

# =============================================================================
# Data Cleaning of Text
# =============================================================================
# Remove Special Characters 
df['Description_Parsed_1'] = df['job.description'].str.replace("\r", " ")
df['Description_Parsed_1'] = df['Description_Parsed_1'].str.replace("\n", " ")
df['Description_Parsed_1'] = df['Description_Parsed_1'].str.replace("\t", " ")
df['Description_Parsed_1'] = df['Description_Parsed_1'].str.replace("\xa0", " ")
df['Description_Parsed_1'] = df['Description_Parsed_1'].str.replace("    ", " ")
# ---------------------------------
# Upercase to Lower 
df['Description_Parsed_2'] = df['Description_Parsed_1'].str.lower()
# ---------------------------------
# Remove Punctuation
punctuation_signs = list("?:!.,;-'®")
df['Description_Parsed_3'] = df['Description_Parsed_2']

for punct_sign in punctuation_signs:
    df['Description_Parsed_3'] = df['Description_Parsed_3'].str.replace(punct_sign, '')
# ---------------------------------
# Remove Possessive Pronouns
df['Description_Parsed_4'] = df['Description_Parsed_3'].str.replace("'s", "")
# ---------------------------------
# Stemming and Lemmatization
nltk.download('punkt')
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

nrows = len(df)
lemmatized_text_list = []

for row in range(0, nrows):
    lemmatized_list = []
    text = df.loc[row]['Description_Parsed_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    lemmatized_text = " ".join(lemmatized_list)
    lemmatized_text_list.append(lemmatized_text)

df['Description_Parsed_5'] = lemmatized_text_list
# ---------------------------------
# Stop Words
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

df['Description_Parsed_6'] = df['Description_Parsed_5']

for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Description_Parsed_6'] = df['Description_Parsed_6'].str.replace(regex_stopword, '')
# ---------------------------------
# Clean Data and Replace Description with Final Parsed Description
list_columns = ["job.title", "company.name", "job.category", "job.subcategory",
                "job.description", "Description_Parsed_6"]
df = df[list_columns]

df = df.rename(columns={'Description_Parsed_6': 'Description_Parsed',
                        'job.title': 'Job_Title',
                        "company.name": 'Company_Name', 
                        "job.category": 'Job_Category',
                        "job.subcategory": 'Job_Subcategory',
                        "job.description": 'Description'})

# df.loc[5]['Description_Parsed']
# df.loc[5]['Description']

# =============================================================================
# Encoding Job Type
# =============================================================================
job_subcategory_codes = {
    'Analysis & Reporting': 0,
    'Analytics': 1,
    'Data Engineering': 2,
    'Data Science': 3
}

# Map Categories to Job Categories
df['Job_Subcategory_Code'] = df['Job_Subcategory']
df = df.replace({'Job_Subcategory_Code':job_subcategory_codes})

# =============================================================================
# Building Training and Test Sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(df['Description_Parsed'], 
                                                    df['Job_Subcategory_Code'], 
                                                    test_size = 0.15, 
                                                    random_state = 42)


# =============================================================================
# Feature Building
# =============================================================================
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 200

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

for Product, category_id in sorted(job_subcategory_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")

bigrams

# =============================================================================
# Save Data Created
# =============================================================================
# X_train
with open('python_ml/pickles/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('python_ml/pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('python_ml/pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('python_ml/pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('python_ml/pickles/df.pickle', 'wb') as output:
    pickle.dump(df, output)
    
# features_train
with open('python_ml/pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('python_ml/pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('python_ml/pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('python_ml/pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('python_ml/pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
# =============================================================================
# =============================================================================
# =============================================================================
