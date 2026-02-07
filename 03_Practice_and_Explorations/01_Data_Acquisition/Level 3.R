# ============================================================
# Pagination
# Level 3 Data Acquisition Project
# Goal: Fetch all posts, but page by page
# ============================================================

library(jsonlite)

# 1. Define the data source
base <- "https://jsonplaceholder.typicode.com/posts"

# 2. Define pagination parameter
all_posts <- list()
max_pages <- 50 #Safeguard
limit <- 10 # We know that there are 100 posts

final_posts <- NULL

# 3.Loop
for (page in 1:max_pages) {
  url <- paste0(base, "?_page=", page, "&_limit=", limit)
  posts <- fromJSON(url)

  # Stop when the API returns nothing
  if (length(posts) == 0) break

  # Combine pages into one dataset
  if (is.null(final_posts)) {
    final_posts <- posts
  } else {
    final_posts <- rbind(final_posts, posts)
  }
}
  
# 4. Print
str(final_posts)
nrow(final_posts)
