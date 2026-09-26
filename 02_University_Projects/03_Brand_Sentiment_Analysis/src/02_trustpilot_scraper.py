import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

url = "https://www.trustpilot.com/review/www.revolut.com"

driver = webdriver.Chrome()
driver.get(url)
time.sleep(10)

html = driver.page_source
print("__NEXT_DATA__" in html)

soup = BeautifulSoup(html, "html.parser")
script = soup.find("script", id="__NEXT_DATA__")
data = json.loads(script.text)
reviews = data["props"]["pageProps"]["reviews"]
print(type(reviews))
print(len(reviews))

driver.quit()