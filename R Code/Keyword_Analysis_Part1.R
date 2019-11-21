# -----------------------------------------------------------------------------
# Script Name: Keyword_Analysis_Part1
#
# Purpose of Script: Initial keyword analysis of job description data
#
# Author: Zachary Richardson, Ph.D.
#
# Date Created: 2019-11-21
#
# Copyright (c) Zachary Richardson, 2019
# Email: zachinthelab@gmail.com
# Blog: thelab.ghost.io
# --------------------------------------
#
# Notes: Perform an initial keyword analysis of the data. This is very basic
#        and needs a lot of revision before being a 'complete' analysis, but
#        it's a really good start.
#
# -----------------------------------------------------------------------------
# Job Keyword Analysis
library(dplyr)
library(stringr)
library(stringi)
library(tm)
library(wordcloud)
library(udpipe)
library(textrank)
library(lattice)
# ----------------------------------------------------------------------------------------
# Load Data
load("full_job_info.RData")
  ds.df <- subset(full_df, job.subcategory == "Data Science")
# ----------------------------------------------------------------------------------------
# Job Keyword Analysis
ud_model <- udpipe_download_model(language = "english-ewt")
ud_model <- udpipe_load_model(ud_model$file_model)
x <- udpipe_annotate(ud_model, x = full_df$job.description)
x <- as.data.frame(x)

x2 <- udpipe_annotate(ud_model, x = ds.df$job.description)
x2 <- as.data.frame(x2)
# ----------------------------------------------------------------------------------------
# Extracting Only Nouns
# ---------------------
stats <- subset(x, upos %in% "NOUN")
stats <- txt_freq(x = stats$lemma)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 30), col = "cadetblue", 
         main = "Most occurring nouns", xlab = "Freq")

stats2 <- subset(x2, upos %in% "NOUN")
stats2 <- txt_freq(x = stats2$lemma)
stats2$key <- factor(stats2$key, levels = rev(stats2$key))
barchart(key ~ freq, data = head(stats2, 30), col = "cadetblue", 
         main = "Most occurring nouns", xlab = "Freq")
# ----------------------------------------------------------------------------------------
# Collocation & Co-Occurrences
# ----------------------------
## Collocation (words following one another)
stats <- keywords_collocation(x = x, term = "token", 
                              group = c("doc_id", "paragraph_id", "sentence_id"),
                              ngram_max = 4)
## Co-occurrences: How frequent do words occur in the same sentence, 
#                  in this case only nouns or adjectives?
stats <- cooccurrence(x = subset(x, upos %in% c("NOUN", "ADJ")), 
                      term = "lemma", group = c("doc_id", "paragraph_id", "sentence_id"))
## Co-occurrences: How frequent do words follow one another?
stats <- cooccurrence(x = x$lemma, 
                      relevant = x$upos %in% c("NOUN", "ADJ"))
## Co-occurrences: How frequent do words follow one another even if we would skip 2 
#                  words in between?
stats <- cooccurrence(x = x$lemma, 
                      relevant = x$upos %in% c("NOUN", "ADJ"), skipgram = 2)
head(stats)

  stats$phrase <- paste0(stats$term1 ," ", stats$term2)
  stats$phrase <- factor(stats$phrase, levels = rev(stats$phrase))
  barchart(phrase ~ cooc, data = head(stats, 30), col = "cadetblue", 
           main = "Most occurring phrases", xlab = "Freq")
  
# ----------------------------------------------------------------------------------------
# Corpus Keyword Analysis
# -----------------------
docs <- Corpus(VectorSource(full_df$job.description))
# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 15)
# ----------------------------------------------------------------------------------------
