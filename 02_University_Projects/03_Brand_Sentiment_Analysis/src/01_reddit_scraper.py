# 0. Import
import requests
import time
import pandas as pd

# 1. Define source variables and config variables
sources = [
    ("revolut", None),
    ("Revolut_EU", None),
    ("neobanks", "revolut"),
    ("banking", "revolut"),
    ("CryptoCurrency", "revolut"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
all_posts = []

# 2. Scrape
for sub, query in sources:
    before = None

    # 2b. Cap the page
    page = 0

    # 2c. 
    while True:
        page += 1
        if page > 40:
            break

        if query is None:
            base_url = f"https://arctic-shift.photon-reddit.com/api/posts/search?subreddit={sub}&limit=100"
        else:
            base_url = f"https://arctic-shift.photon-reddit.com/api/posts/search?query={query}&subreddit={sub}&limit=100"

        if before:
            url = f"{base_url}&before={before}"
        else:
            url = base_url

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code} on r/{sub}")
            break

        try:
            data = response.json()
        except Exception as e:
            print(f"Bad response from r/{sub}: {e}")
            break

        posts = data["data"]

        if not posts:
            break

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

        print(f" r/{sub}: fetched {len(posts)} posts (total: {len(all_posts)})")
        before = posts[-1]["created_utc"]
        time.sleep(2)


print(f"Total posts collected: {len(all_posts)}")

# 3. Clean and save files
df = pd.DataFrame(all_posts)
df.to_csv("data/raw/reddit_data.csv", index=False)
print(f"file succesfully downloaded!")