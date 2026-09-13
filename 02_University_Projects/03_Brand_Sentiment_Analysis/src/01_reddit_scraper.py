# 0. Import
import requests
import time
import pandas as pd

# 1. Define source variables and config variables
sources = [
    ("revolut", None),
    ("fintech", "revolut"),
    ("personalfinance", "revolut"),
    ("UKPersonalFinance", "revolut"),
    ("eupersonalfinance", "revolut"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
all_posts = []

# 2. Scrape
for sub, query in sources:
    if query is None:
        base_url = f"https://arctic-shift.photon-reddit.com/api/posts/search?subreddit={sub}&limit=100"
    else:
        base_url = f"https://arctic-shift.photon-reddit.com/api/posts/search?query={query}&subreddit={sub}&limit=100"

    response = requests.get(base_url, headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code} on r/{sub}")
        continue

    try:
        data = response.json()
    except Exception as e:
        print(f"Bad response from r/{sub}: {e}")
        continue

    posts = data["data"]
    for p in posts:
        all_posts.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "text": p.get("selftext", ""),
            "score": p.get("score", 0),
            "num_comments": p.get("num_comments", 0),
            "created_utc": p.get("created_utc", ""),
            "author": p.get("author", ""),
            "subreddit": p.get("subreddit", sub),
        })

    time.sleep(2)


print(f"Total posts collected: {len(all_posts)}")

# 3. Clean and save files
df = pd.DataFrame(all_posts)
df.to_csv("data/raw/reddit_data.csv", index=False)
print(f"file succesfully downloaded!")