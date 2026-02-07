# ============================================================
# Semi-structured data handling
# Level 6 Data Acquisition Project
# Goal: Get posts, then enrich them with user information (author name/email).
# ============================================================

library(jsonlite)

posts <- fromJSON("https://jsonplaceholder.typicode.com/posts")
users <- fromJSON("https://jsonplaceholder.typicode.com/users")

ensure_col <- function(df, col) {
  if (!col %in% names(df)) {
    df[[col]] <- NA
  }
  df
}

# Defensive checks
posts <- ensure_col(posts, "userId")
posts <- ensure_col(posts, "id")

users <- ensure_col(users, "id")
users <- ensure_col(users, "name")

# Normalize types
posts$userId <- as.integer(posts$userId)
users$id     <- as.integer(users$id)

# Safe join
posts_enriched <- merge(
  posts,
  users[, c("id", "name")],
  by.x = "userId",
  by.y = "id",
  all.x = TRUE
)

str(posts_enriched)
