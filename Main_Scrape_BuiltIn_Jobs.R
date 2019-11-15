# -----------------------------------------------------------------------------
# Script Name: Main_Scrape_BuiltIn_Jobs
#
# Purpose of Script: Scrape job description data to use in analysis from 
#                    https://www.builtincolorado.com for data science positions
#
# Author: Zachary Richardson, Ph.D.
#
# Date Created: 2019-11-14
#
# Copyright (c) Zachary Richardson, 2019
# Email: zachinthelab@gmail.com
# Blog: thelab.ghost.io
# --------------------------------------
#
# Notes: Basic scrape of key information from Built In from the Colorado job
#        board. This will focus on the 'Data Analysis' set of jobs but could
#        expanded to include all of the jobs that are listed.
#
# -----------------------------------------------------------------------------
# Load Packages 
library(rvest)
library(XML)
library(dplyr)
library(stringr)
library(stringi)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Start with links to the primary job board and the data analytics job board specifically
alljobs.url <- "https://www.builtincolorado.com/jobs"
datajobs.url <- "https://www.builtincolorado.com/jobs?f[0]=job-category_data-analytics"
# ----------------------------------------------------------------------------------------
# First use Selector Gadget to get the URLs all the pages. (Note: Built In uses a zero
# page number format so if there are 6 pages then the max page number will be 5 so it
# seems more efficient to pull the links than to pull the values.)
page.links <- read_html(datajobs.url) %>%
  html_nodes('.js-pager__items') %>%
  html_nodes('a') %>%
  html_attr('href') %>%
  unique()
# One other note, I choose to mostly use CSS at first for the selector items since it's
# shorter and easier to identify exactly wher ewe want to look at things.
# --------------------------------------
# --------------------------------------
# Now itterate over the links to pull in the key info for each job description:
full_df <- data.frame()
for(i in page.links) {
  url <- paste0(alljobs.url, i)
  page <- read_html(url)

  # get job titles
  job.title <- page %>%
    html_nodes('div.block-views-blockjobs-jobs-landing') %>%
    html_nodes('h2') %>%
    html_text()
    
  # get links
  links <- page %>% 
    html_nodes('div.block-views-blockjobs-jobs-landing') %>%
    html_nodes(xpath = ".//div[@class='view-content']") %>%
    html_nodes(xpath = ".//div[@class='wrap-view-page']") %>%
    html_nodes('a') %>%
    html_attr('href')
  
  job.description <- c()
  for(i in seq_along(links)) {
    
    url <- paste0("https://www.builtincolorado.com",links[i])
    page <- read_html(url)
    
    job.description[[i]] <- page %>%
      html_nodes(xpath = "//div[@class='node__content']") %>%
      html_text() %>%
      str_replace_all(., '[\r\n]' , ' ') %>%
      stri_trim_both()
  }
  
  company.name <- c()
  for(i in seq_along(links)) {
    
    url <- paste0("https://www.builtincolorado.com",links[i])
    page <- read_html(url)
    
    company.name[[i]] <- page %>%
      html_nodes('div.job-info-wrapper') %>%
      html_nodes(xpath = ".//div[@class='field__item']") %>%
      html_text() %>%
      stri_trim_both()
  }  
  df <- data.frame(job.title, company.name, job.description)
  full_df <- rbind(full_df, df)
}
full_df$job.title <- as.character(full_df$job.title)
full_df <- unique(full_df)
# ----------------------------------------------------------------------------------------
save(full_df, file = "full_job_info.RData")
# ----------------------------------------------------------------------------------------
