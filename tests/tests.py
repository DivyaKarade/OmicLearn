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

# Define
sample_df = chrome.find_element_by_class_name('block-container')
print(sample_df)

sample = chrome.find_element_by_xpath("//div[@class='st-cn st-dh st-dd st-cg st-bo st-bp st-bq st-br st-bt st-ax st-b0 st-c4 st-c6 st-db st-c5 st-cb st-cc st-di st-d9 st-dj st-dk']")
print(sample)

trial = chrome.find_element_by_xpath("//div[@class='Widget row-widget stSelectbox']")
print(trial)

# Quit Driver
chrome.quit()
