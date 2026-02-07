# ============================================================
# Failure-aware Pipeline
# Level 7 Data Acquisition Project
# Goal: Get posts, then enrich them with user information (author name/email).
# ============================================================

library(jsonlite)

base <- "https://jsonplaceholder.typicode.com/posts"
limit <- 10
max_pages <- 50

final_posts <- NULL
log <- character(0)

safe_fetch <- function(url) {
  x <- try(fromJSON(url), silent = TRUE)
  if (inherits(x, "try-error")) return(NULL)
  x
}

for (page in 1:max_pages) {
  url <- paste0(base, "?_page=", page, "&_limit=", limit)
  
  posts <- safe_fetch(url)
  
  # Retry once if failed
  if (is.null(posts)) {
    log <- c(log, paste("Page", page, "failed. Retrying..."))
    Sys.sleep(1)
    posts <- safe_fetch(url)
  }
  
  # If still failed, log and CONTINUE (don’t crash)
  if (is.null(posts)) {
    log <- c(log, paste("Page", page, "failed twice. Skipping."))
    next
  }
  
  # Stop condition: no more data
  if (length(posts) == 0) {
    log <- c(log, paste("Page", page, "returned 0 rows. Done."))
    break
  }
  
  # Combine
  if (is.null(final_posts)) final_posts <- posts
  else final_posts <- rbind(final_posts, posts)
  
  Sys.sleep(0.2)
}

# Output results
if (!is.null(final_posts)) {
  message("Rows collected: ", nrow(final_posts))
} else {
  message("No rows collected.")
}

# Print log
cat(paste(log, collapse = "\n"), "\n")