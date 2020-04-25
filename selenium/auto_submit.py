'''
Automatically submit URLS and Titles for analysis. 
Might need to wait several minutes for analysis to be completed (card will appear under 'Needs Review' section)

'''

import time
from selenium import webdriver
from time import sleep

# * Connect to Chrome and local URL
driver = webdriver.Chrome('../chromedriver')       # Optional argument, if not specified will search path.
driver.get('http://localhost:9999/')

# * Input URL(s) for analysis
url_input = driver.find_element_by_id('url')          # print(elem), elem.click()
url_input.send_keys("https://blog.cloudflare.com/inside-mirai-the-infamous-iot-botnet-a-retrospective-analysis/")

# * Input URLs' corresponding titles to keep track more easily
title_input = driver.find_element_by_id('title')
title_input.send_keys('https://blog.cloudflare.com/inside-mirai-the-infamous-iot-botnet-a-retrospective-analysis/')

# * Once both inputs filled, submit form and await results
submit_btn = driver.find_element_by_css_selector("button[onclick='submit_report()']")
submit_btn.click()
print('SUBMITTED')

# * Buffer time to allow sufficient time for submission before quitting
sleep(5)
driver.quit()