from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.keys import Keys

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.binary_location = '/usr/bin/google-chrome-stable'
chrome_driver = '/usr/bin/chromedriver'

driver = webdriver.Chrome(executable_path=chrome_driver, chrome_options=chrome_options)
driver.get('https://box.attie.co.uk/')

assert "my box..." in driver.page_source
driver.close()
