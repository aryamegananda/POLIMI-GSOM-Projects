import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

url = "https://www.trustpilot.com/review/www.revolut.com"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = 

print(response.status_code)
print(response.text[:500])

soup = BeautifulSoup(response.text, "html.parser")
script = soup.find("script", id="__NEXT_DATA__")
data = json.loads(script.text)

print(response.status_code)

