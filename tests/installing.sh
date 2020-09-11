# apt update && apt upgrade -y
apt update

# install dependencies
apt install -y curl gpg wget python3-pip virtualenv unzip

# install chrome
# curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add
# echo "deb [arch=amd64]  http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
# apt update
# apt install -y google-chrome-stable
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt install ./google-chrome-stable_current_amd64.deb
apt-get install -y chromium-browsera

# install chromedriver
# see: https://sites.google.com/a/chromium.org/chromedriver/downloads
cd /tmp
wget https://chromedriver.storage.googleapis.com/76.0.3809.68/chromedriver_linux64.zip
unzip /tmp/chromedriver_linux64.zip
mv chromedriver /usr/bin/chromedriver
chown root: /usr/bin/chromedriver
chmod 777 /usr/bin/chromedriver

# install selenium client (python)
# virtualenv -p python3.6 venv
# source venv/bin/activate
pip3 install selenium