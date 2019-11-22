# -----------------------------------------------------------------------------
# Script Name: Scrape_Field_Nodes_BiC
#
# Purpose of Script: Scrape technical tools from https://www.builtincolorado.com 
#                    for different data subsets.
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
# Notes: Scrape the "Technology We Use" field for each job listed in the
#        Data + Analytics category as well as other job subcategories.
#
# -----------------------------------------------------------------------------
# Load Packages 
library(rvest)
library(XML)
library(dplyr)
library(tidyverse)
library(stringr)
library(stringi)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Starter links needed:
basejobs.url <- "https://www.builtincolorado.com/jobs"
alldata.urls <- "https://www.builtincolorado.com/jobs?f[0]=job-category_data-analytics"
dsonly.urls <-  "https://www.builtincolorado.com/jobs?f[0]=job-category_data-analytics-data-science&&hash-changes=4"
anydata.urls <- "https://www.builtincolorado.com/jobs?f[0]=job-category_data-analytics-analysis-reporting&f[1]=job-category_data-analytics-analytics&f[2]=job-category_data-analytics-data-science&f[3]=job-category_data-analytics-machine-learning&hash-changes=7"
# ----------------------------------------------------------------------------------------
# Pull pagination links for three different types of pulls:
#  1. Any Data + Analytics Jobs
#  2. Data Science Only Links
#  3. Jobs classified as: Analysis & Reporting, Analytics, Data Science, & Machine Learning
# --------------------------------------
# Same as before, build pages first then build links to every job by topic
# --------------------------------------
#  1. Any Data + Analytics Jobs
alldata.links <- read_html(alldata.urls) %>%
  html_nodes('.js-pager__items a') %>%
  html_attr('href') %>%
  unique()
# --------------------------------------
#  2. Data Science Only Links
dsonly.links <- read_html(dsonly.urls) %>%
  html_nodes('.js-pager__items a') %>%
  html_attr('href') %>%
  unique()
# --------------------------------------
#  3. Data 4 Categories Links
anydata.links <- read_html(anydata.urls) %>%
  html_nodes('.js-pager__items a') %>%
  html_attr('href') %>%
  unique()
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Simplifying the repetitive scrapes by creating custom function:
joblinks <- function(link.list) {
  job.links <- c()
  for(i in link.list) {
    url <- paste0(basejobs.url, i)
    page <- read_html(url)
    # get links
    links <- page %>%
      html_nodes('.block-views-blockjobs-jobs-landing .wrap-view-page a') %>%
      html_attr('href')
    links <- paste0("https://www.builtincolorado.com", links)
    job.links <- append(job.links, links)
  } 
  job.links <- unique(job.links)
  return(job.links)
}
# --------------------------------------
jobtech <- function(link.list) {
  job.links <- c()
  for(i in link.list) {
    url <- paste0(basejobs.url, i)
    page <- read_html(url)
    # get links
    links <- page %>%
      html_nodes('.block-views-blockjobs-jobs-landing .wrap-view-page a') %>%
      html_attr('href')
    links <- paste0("https://www.builtincolorado.com", links)
    job.links <- append(job.links, links)
  } 
  job.links <- unique(job.links)

    tech.item <- c()
    for(i in job.links) {
      page <- read_html(i)
      itm.temp <- page %>%
        html_nodes('.block-bix-companies-our-full-stack-block 
                    .tab-content li .full-stack-item') %>%
        html_text()
      
      itm.temp <- if(length(itm.temp) > 0) 
        {itm.temp <- itm.temp} 
        else{itm.temp <- NA}
      
      tech.item <- append(tech.item, itm.temp)
    }
    tech.field <- c()
    subcats <- c()
    jlink <- c()
    for(i in job.links) {
      page <- read_html(i)
      fld.temp <- page %>%
        html_nodes('.block-bix-companies-our-full-stack-block 
                    .tab-content li .full-stack-item-field') %>%
        html_text()
      
      fld.temp <- if(length(fld.temp) > 0) 
        {fld.temp <- fld.temp} 
        else{fld.temp <- NA}
    
      sub.temp <- page %>%
        html_nodes('.job-category-links a:nth-child(3)') %>%
        html_text() %>%
        str_replace_all(., '[\r\n]' , ' ') %>%
        stri_trim_both()
      
      sub.temp <- tryCatch(
        c(sub.temp, rep(NA, (length(fld.temp)-1))),
        error=function(e){NA})
         
      job.temp <- i
        job.temp <- tryCatch(
        c(job.temp, rep(NA, (length(fld.temp)-1))),
        error=function(e){NA})
      
      tech.field <- append(tech.field, fld.temp)
      subcats <- append(subcats, sub.temp)
      jlink <- append(jlink, job.temp)
    }
  df <- data.frame(tech.item, tech.field)
    subcats <- data.frame(subcats, stringsAsFactors = FALSE) %>% fill(subcats)
    jlink <- data.frame(jlink, stringsAsFactors = FALSE) %>% fill(jlink)
  df <- cbind(df, subcats, jlink)
  colnames(df) <- c('tech.item', 'tech.field', 'list.type', 'job.link')
    df <- na.omit(df)
  return(df)
}
# --------------------------------------
#  1. Any Data + Analytics Jobs
alldata.techdf <- jobtech(alldata.links)
#  2. Data Science Only Links
dsonly.techdf <- jobtech(dsonly.links)
#  3. Data 4 Categories Links
anydata.techdf <- jobtech(anydata.links)
# ----------------------------------------------------------------------------------------
save(alldata.techdf, dsonly.techdf, anydata.techdf , file = "techdfs.RData")
# ----------------------------------------------------------------------------------------
