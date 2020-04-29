'''
Go to analysed report's URL and click 'Export to PDF and save to database'

'''

import argparse
from selenium import webdriver
from time import sleep

# * Initialise parser
parser = argparse.ArgumentParser(
    description="Submit URLS and their corresponding Titles (each comma-separated) to TRAM"
)

# * Add positional/optional arguments (short form, long form, help text, default value)
parser.add_argument('titlefile', help="path to file containing comma-separated titles")

# * Parse arguments
args = parser.parse_args()
titlefile = args.titlefile

with open(titlefile, 'r') as f:
    content = f.read()

titles = content.split(',')
# for t in titles:
#     print(not t)
#     print((f'http://localhost:9999/edit/{t}'))

driver = webdriver.Chrome('../chromedriver') 

for title in titles:
    if not title:
        continue

    driver.get(f'http://localhost:9999/edit/{title}')
    # pdf_elem = driver.find_element_by_css_selector("div[class='col-md-auto']")
    pdf_elem = driver.find_element_by_class_name("btn-outline-secondary")
    print(pdf_elem.text)
    pdf_elem.click()
    sleep(10)



# driver.quit()
