'''
Pass as arguments files containing URLS and corresponding Titles; automatically submit them for analysis. 
Might need to wait several minutes for analysis to be completed (card will appear under 'Needs Review' section)

'''

import time
import argparse
from selenium import webdriver
from time import sleep

# * Initialise parser
parser = argparse.ArgumentParser(
    description="Submit URLS and their corresponding Titles (each comma-separated) to TRAM"
)

# * Add positional/optional arguments (short form, long form, help text, default value)
parser.add_argument('urlfile', help="path to file containing comma-separated urls")
parser.add_argument('titlefile', help="path to file containing comma-separated titles")

# * Parse arguments
args = parser.parse_args()
urlfile = args.urlfile
titlefile = args.titlefile

with open(urlfile, 'r') as uf:
    URLS = uf.read()
print(URLS, type(URLS))

with open(titlefile, 'r') as tf:
    TITLES = tf.read()
print(TITLES, type(TITLES))


# * Connect to Chrome and local URL
driver = webdriver.Chrome('../../chromedriver')       # Optional argument, if not specified will search path.
driver.get('http://localhost:9999/')

# * Input URL(s) for analysis
url_input = driver.find_element_by_id('url')          # print(elem), elem.click()
url_input.send_keys(URLS)

# * Input URLs' corresponding titles to keep track more easily
title_input = driver.find_element_by_id('title')
title_input.send_keys(TITLES)

# * Once both inputs filled, submit form and await results
submit_btn = driver.find_element_by_css_selector("button[onclick='submit_report()']")
submit_btn.click()
print('SUBMITTED')

# * Buffer time to allow sufficient time for submission before quitting
# sleep(5)
# driver.quit()

