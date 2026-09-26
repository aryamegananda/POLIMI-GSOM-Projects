import pandas as pd
import numpy as np

# 1. You receive a messy CSV. Write a function that takes a filepath and prints: 
# number of rows, number of columns, column names, and count of null values per column.

def explore(filepath):
    df = pd.read_csv(filepath)
    print(len(df)) #print number of rows
    print(len(df.columns)) #print number of columns
    print(df.columns) #print column names
    print(df.isnull().sum()) #print count of null values
    return df

explore("vimodrone_results.csv")


# 2. Write a function that takes a DataFrame and returns it with all column names lowercased, 
# spaces replaced with underscores, and leading/trailing whitespace stripped. 
# So " First Name " becomes "first_name".

def clean_columns(df):
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.strip()
    return df


# 3. Write a function that takes a DataFrame, a column name, and a multiplier. 
# Flag any value more than multiplier standard deviations from the mean as True in a new column called is_outlier.


def flag_outliers(df,col,mul):
    mean = df[col].mean()
    treshold = mul * df[col].std()
    df["is_outlier"] = (abs(df[col]-mean) > treshold)
    return df


# 4. You get a column with mixed date formats: 
# "2025-01-15", "15/01/2025", "January 15, 2025". Write a function that parses all of them into proper datetime.
def parse_mixed_dates(df, col):
    df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True)
    return df


# 5. A DataFrame has duplicate sample_id rows. 
# When duplicates exist, keep the row with status = 'passed' over 'pending' over 'failed'. Write a function that does this.
def smart_dedup(df):
    priority = {
        "passed": 1,
        "pending": 2,
        "failed": 3
    }

    df["priority"] = df["status"].map(priority)
    df = df.sort_values(df["priority"], ascending=True)
    df = df.drop_duplicates(subset="sample_id", keep="first")
    df = df.drop(columns=["priority"])
    return df


# 6. Write a function that takes the test_results DataFrame 
# and returns a pivot table showing: rows = site_code, columns = status, values = count. With a total column.

def status_summary(df):
    summary = pd.pivot_table(df, values="sample_id", index="site_code", columns="status", aggfunc="count", margins=True)
    return summary