from bs4 import BeautifulSoup
import requests
import pandas as pd
import re


URL  = "https://www2.hm.com/en_us/search-results.html?q=TSHIRT"
# Headers for request
HEADERS = ({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36', 
 'Accept-Language': 'da, en-gb, en',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Referer': 'https://www.google.com/'
}) #add your user agent 
# HTTP Request
webpage = requests.get(URL, headers=HEADERS)

# print(webpage)
# Soup Object containiang all data
soup = BeautifulSoup(webpage.content, "html.parser")

#To find <a class>
# with open("output1.html", "w", encoding='utf-8') as file:
#     file.write(str(soup))

links = soup.find_all('a', class_='link', href=lambda href: href and 'productpage' in href)


###############Individual product pages############
link = links[0].get('href')
product_list = "https://www2.hm.com" + link
new_webpage = requests.get(product_list, headers=HEADERS)
new_soup = BeautifulSoup(new_webpage.content, "html.parser")
img_tag = new_soup.find('img', attrs={'srcset': True})
# Extract the srcset attribute value
srcset_value = img_tag.get('srcset')

# Define the pattern to search for
pattern = r'(url\[file:/product/main\]).*'

# Use re.sub() to perform the replacement
modified_string = re.sub(pattern, r'\1', srcset_value)

# Remove repeating lines
lines = modified_string.strip().split('\n')
unique_lines = list(dict.fromkeys(lines))

# Construct the product link
product_img_link = 'https:' + unique_lines[0] 
print(product_img_link)
