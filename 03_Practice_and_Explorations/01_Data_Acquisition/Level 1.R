# ============================================================
# Daily Exchange Rates Snapshot
# Level 1 Data Acquisition Project
# Goal: What are today’s exchange rates relative to EUR?
# ============================================================

library(jsonlite)

# Step 1: Define the data source
url <- "https://raw.githubusercontent.com/aryamegananda/POLIMI-GSOM-Projects/main/03_Practice_and_Explorations/01_Data_Acquisition/Dataset/exchange_rates.json"


# Step 2: Parse JSON into R
data <- fromJSON(url)

# Step 3: Inspect the data
str(data)

# Step 4: Print
data