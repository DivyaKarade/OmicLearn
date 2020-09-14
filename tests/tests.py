from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options

# Set Chrome WebDriver and Configurate It  
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome = webdriver.Chrome(desired_capabilities=DesiredCapabilities.CHROME, chrome_options=chrome_options)

# Start Testing
endpoint = "http://ec2-3-121-216-179.eu-central-1.compute.amazonaws.com:8501/"
chrome.get(endpoint)
print(chrome.title)

# Quit Driver
chrome.quit()
