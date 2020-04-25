'''
Go to analysed report's URL and export to PDF

'''

import time
from selenium import webdriver
from time import sleep

driver = webdriver.Chrome('../../chromedriver') 
driver.get('http://localhost:9999/edit/Title')

# pdf_elem = driver.find_element_by_css_selector("div[class='col-md-auto']")
pdf_elem = driver.find_element_by_class_name("btn-outline-secondary")
print(pdf_elem.text)
pdf_elem.click()
sleep(5)
driver.quit()
