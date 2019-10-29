#Loading the rvest package
library(rvest)
library(XML)
library(dplyr)
library(stringr)
library(stringi)
# ----------------------------------------------------------------------------------------
# all.job.html <- 'https://www.builtincolorado.com/jobs?f[0]=job-category_data-analytics&hash-changes=2&page=0'
all.job.html <- "https://www.builtincolorado.com/jobs?f[0]=job-category_data-analytics&f[1]=job-category_data-analytics-analytics&f[2]=job-category_data-analytics-data-science&hash-changes=7"

page_result_start <- 0 # starting page 
page_result_end <- 2 # last page results
page_results <- seq(from = page_result_start, to = page_result_end, by = 1)
 
full_df <- data.frame()
for(i in seq_along(page_results)) {
  first_page_url <- "https://www.builtincolorado.com/jobs?f[0]=job-category_data-analytics&hash-changes=2&page="
  url <- paste0(first_page_url, page_results[i])
  page <- read_html(url)
  # Sys.sleep(2)

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
# ----------------------------------------------------------------------------------------
setwd("/Users/Admin/Dropbox/Personal Research Files/Job Post Keyword Searches")
save(full_df, file = "full_job_info.RData")
# ----------------------------------------------------------------------------------------
