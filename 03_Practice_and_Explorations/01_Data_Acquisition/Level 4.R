# ============================================================
# Pagination with Temporary Failures
# Level 4 Data Acquisition Project
# Goal: Fetch all posts, but page by page, and anticipate fail request
# ============================================================

library(jsonlite)

# 1. Define the data source
base <- "https://jsonplaceholder.typicode.com/posts"

# 2. Define pagination variables
limit <- 10
max_pages <- 50

final_posts <- NULL

# 3. Loop
for (page in 1:max_pages) {
  
  url <- paste0(base, "?_page=", page, "&_limit=", limit)
  
  posts <- try(fromJSON(url), silent = TRUE)
  
  # If request failed, retry once after waiting
  if (inherits(posts, "try-error")) {
    Sys.sleep(2)
    posts <- try(fromJSON(url), silent = TRUE)
  }
  
  # If still nothing, stop
  if (inherits(posts, "try-error") || length(posts) == 0) {
    break
  }
  
  # Combine
  if (is.null(final_posts)) {
    final_posts <- posts
  } else {
    final_posts <- rbind(final_posts, posts)
  }
  
  # Be polite to the API
  Sys.sleep(0.5)
}

str(final_posts)
nrow(final_posts)
