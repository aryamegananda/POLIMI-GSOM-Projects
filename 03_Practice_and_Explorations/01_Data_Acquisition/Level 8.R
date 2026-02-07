# ============================================================
# Provenance, snapshots, and reproducibility
# Level 8 Data Acquisition Project
# Goal: Get posts, then enrich them with user information (author name/email).
# ============================================================

library(jsonlite)

# 1. Define data source & pagination
base <- "https://jsonplaceholder.typicode.com/posts"

limit <- 10
max_pages <- 50

# 2. Container
final_posts <- NULL
log <- character(0)

# 3. Loop
for (page in 1:max_pages) {
  
  url <- paste0(base, "?_page=", page, "&_limit=", limit)
  
  posts <- try(fromJSON(url), silent = TRUE)
  
  # Retry once if failed
  if (inherits(posts, "try-error")) {
    log <- c(log, paste("Page", page, ": first attempt failed"))
    Sys.sleep(1)
    posts <- try(fromJSON(url), silent = TRUE)
  }
  
  # If still failed → skip but continue
  if (inherits(posts, "try-error")) {
    log <- c(log, paste("Page", page, ": skipped after retry"))
    next
  }
  
  # Stop when no more data
  if (length(posts) == 0) {
    log <- c(log, paste("Page", page, ": no more data, stopping"))
    break
  }
  
  # Combine pages
  if (is.null(final_posts)) {
    final_posts <- posts
  } else {
    final_posts <- rbind(final_posts, posts)
  }
  
  Sys.sleep(0.3)
}

# 4. Create output folders
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

# 5. Save processed data
write.csv(
  final_posts,
  "data/processed/posts.csv",
  row.names = FALSE
)

#6. Save run metadata
run_metadata <- list(
  run_time = Sys.time(),
  source_url = base,
  rows_collected = if (!is.null(final_posts)) nrow(final_posts) else 0,
  pages_attempted = page,
  notes = "Beginner Level-8 data acquisition run"
)

writeLines(
  toJSON(run_metadata, pretty = TRUE, auto_unbox = TRUE),
  "data/run_metadata.json"
)

#7. Result inspection
str(final_posts)
nrow(final_posts)

cat("\nLOG:\n")
cat(paste(log, collapse = "\n"))