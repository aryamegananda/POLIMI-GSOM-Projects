
library(jsonlite)

# Step 1: Define the data source (fixed URL)
url <- "https://jsonplaceholder.typicode.com/users"

# Step 2: Acquire + parse the data
data <- fromJSON(url)

# Step 3: Inspect the structure
str(data)
