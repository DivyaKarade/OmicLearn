
# Basics
apt update
apt install -y curl wget python3-pip virtualenv unzip

# Install driver
cd ~
mkdir .local
cd .local
mkdir bin
cd bin
wget https://github.com/mozilla/geckodriver/releases/download/v0.19.1/geckodriver-v0.19.1-linux64.tar.gz
tar xvfz geckodriver-v0.19.1-linux64.tar.gz
# mv geckodriver ~/.local/bin

# Install selenium
pip install selenium
pip3 install selenium


