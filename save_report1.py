'''
Go to analysed report's URL (handles only ONE) and click 'Export to PDF and save to database'

'''

import argparse
from selenium import webdriver
from time import sleep

# * Initialise parser
parser = argparse.ArgumentParser(
    description="Submit URLS and their corresponding Titles (each comma-separated) to TRAM"
)

# Add required argument and parse
parser.add_argument('title', help="Title")
args = parser.parse_args()
title= args.title

driver = webdriver.Chrome('../chromedriver') 
driver.get(f'http://localhost:9999/edit/{title}')
pdf_elem = driver.find_element_by_class_name("btn-outline-secondary")
print(pdf_elem.text)
pdf_elem.click()

# sleep(5)
# driver.quit()
