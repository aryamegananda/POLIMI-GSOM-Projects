# ============================================================
# Multiple endpoint acquisition + joining
# Level 5 Data Acquisition Project
# Goal: Get posts, then enrich them with user information (author name/email).
# ============================================================

library(jsonlite)

# 1. Fetch posts (Dataset A)
posts_url <- "https://jsonplaceholder.typicode.com/posts"
posts <- fromJSON(posts_url)

# 2. Fetch users (Dataset B)
users_url <- "https://jsonplaceholder.typicode.com/users"
users <- fromJSON(users_url)

# 3. Select only the user columns we need (keep it simple)
users_small <- users[, c("id", "name", "email")]

# 4. Join: add user info to each post
# posts$userId matches users_small$id
posts_enriched <- merge(
  x = posts,
  y = users_small,
  by.x = "userId",
  by.y = "id",
  all.x = TRUE
)

# 5. Inspect result
str(posts_enriched)
head(posts_enriched[, c("userId", "id", "title", "name", "email")])
