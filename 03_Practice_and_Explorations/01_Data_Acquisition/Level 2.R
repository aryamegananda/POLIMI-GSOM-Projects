# ============================================================
# Parameterized Data Acquisition
# Level 2 Data Acquisition Project
# Goal: Fetch users, posts, and posts for a specific user — by changing parameters.
# ============================================================

library(jsonlite)

# 1. Define the data source
base <- "https://jsonplaceholder.typicode.com"

# 2. Fetch all users (no parameters)
users_url <- paste0(base, "/users")
users <- fromJSON(users_url)
str(users)

# 2) Fetch all posts (no parameters)
posts_url <- paste0(base, "/posts")
posts <- fromJSON(posts_url)
str(posts)

# 3) Fetch posts for ONE user (parameter)
user_id <- 3
posts_user_url <- paste0(base, "/posts?userId=", user_id)
posts_user <- fromJSON(posts_user_url)
str(posts_user)

# Quick sanity checks
nrow(posts)        # should be 100
nrow(posts_user)   # should be 10 (for each userId)
unique(posts_user$userId)