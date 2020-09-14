from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

chrome = webdriver.Chrome(desired_capabilities=DesiredCapabilities.CHROME, chrome_options=chrome_options)

chrome.get('https://www.google.com')
print(chrome.title)

chrome.quit()
