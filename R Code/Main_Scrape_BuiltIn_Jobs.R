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
  html_nodes('.js-pager__items a') %>%
  html_attr('href') %>%
  unique()
# One other note, I choose to mostly use CSS at first for the selector items since it's
# shorter and easier to identify exactly wher ewe want to look at things.
# --------------------------------------
# --------------------------------------
# Now iterate over the links to pull in the key info for each job description:
job.links <- c()
for(i in page.links) {
  url <- paste0(alljobs.url, i)
  page <- read_html(url)

  # get links
  links <- page %>%
    html_nodes('.block-views-blockjobs-jobs-landing .wrap-view-page a') %>%
    html_attr('href')
  links <- paste0("https://www.builtincolorado.com", links)
  job.links <- append(job.links, links)
} 
job.links <- unique(job.links)
# -------------------------------------------------
job.title <- c()
company.name <- c()
job.description <- c()
job.category <- c()
job.subcategory <- c()
# ---------------------
for(i in job.links) {
  page <- read_html(i)
  
  # get job titles
  title <- page %>%
    html_nodes('.job-info-wrapper h1') %>%
    html_text() %>%
    str_replace_all(., '[\r\n]' , ' ') %>%
    stri_trim_both()
  job.title <- append(job.title, title)
  
  # get company names
  c.name <- page %>%
    html_nodes('.job-info a') %>%
    html_text() %>%
    str_replace_all(., '[\r\n]' , ' ') %>%
    stri_trim_both()
  company.name <- append(company.name, c.name)

  # get job descriptions
  j.desc <- page %>%
    html_nodes(xpath = "//div[@class='job-description fade-out']") %>%
    html_text() %>%
    str_replace_all(., '[\r\n]' , ' ') %>%
    stri_trim_both()
  job.description <- append(job.description, j.desc)

  cat <- page %>%
    html_nodes('.job-category-links a:nth-child(1)') %>%
    html_text() %>%
    str_replace_all(., '[\r\n]' , ' ') %>%
    stri_trim_both()
  job.category <- append(job.category, cat)
  
  subcat <- page %>%
    html_nodes('.job-category-links a:nth-child(3)') %>%
    html_text() %>%
    str_replace_all(., '[\r\n]' , ' ') %>%
    stri_trim_both()
  job.subcategory <- append(job.subcategory, subcat)
  
}
full_df <- data.frame(job.title, company.name, job.category, 
                      job.subcategory, job.description)
  full_df <- full_df %>% mutate_if(is.factor, as.character) %>% unique()

full_df <- unique(full_df)
# ----------------------------------------------------------------------------------------
save(full_df, file = "full_job_info.RData")
# ----------------------------------------------------------------------------------------
