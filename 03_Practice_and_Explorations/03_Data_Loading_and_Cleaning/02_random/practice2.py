import pandas as pd
import numpy as np

# 1. Write a function that takes a DataFrame and a dictionary of 
# {column: type} and converts each column to that type. Skip columns that fail conversion.
def enforce_types(df, type_map):
    for col,dtype in type_map.items():
        try:
            df[col] = df[col].astype(dtype)
        except:
            pass
    return df


# **Exercise 8: Value replacer**
# Write a function that takes a DataFrame, a column name, and a mapping dictionary. 
# Replace all values in that column using the mapping. Any value not in the mapping stays as-is (don't lose it).

def map_values(df, col, mapping):
    try:
        df[col] = df[col].replace(mapping)
    except:
        pass
    return df

# **Exercise 9: Null report**
# Write a function that takes a DataFrame and returns a new DataFrame showing: column name, null count, null percentage, and data type. 
# Sorted by null percentage descending.

def null_report(df):
    null = df.isnull().sum()
    data = {
        "column_name": df.columns,
        "null_count": null,
        "null_percentage": (null/len(df))*100,
        "data_type": df.dtypes
    }

    new_df = pd.DataFrame(data)
    new_df = new_df.sort_values("null_percentage", ascending=False)
    return new_df

# **Exercise 10: Row filter with logging**

# Write a function that filters out rows where a given column's value is not in an allowed list. Print how many rows were removed. 
# Return the cleaned DataFrame.

def filter_valid(df, col, allowed):
    before = len(df)
    df = df[df[col].isin(allowed)]
    after = len(df)
    print(f"Removed {before-after} rows")
    return df


# **Exercise 11: Multi-site merger**

# Write a function that takes a dictionary of `{site_name: DataFrame}`, adds a `site_code` column to each, 
# and combines them all into one DataFrame.

site_data = {"vimodrone": df1, "poggibonsi": df2, "milan_hq": df3}
site_codes = {"vimodrone": "VIM", "poggibonsi": "POG", "milan_hq": "MIL"}



# **Exercise 12: Pipeline logger**

# Write a function decorator that wraps any transform function. It prints the function name, row count before, row count after, and how many rows changed.

# ```python
# def log_step(func):
#     # your code here

# @log_step
# def remove_duplicates(df):
#     return df.drop_duplicates()
# ```

# All ETL patterns. Try them — these are the exact functions you'd write at Eurofins.