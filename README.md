# Create a virtual environment
# This has been tested in Ubuntu or Windows WSL
## Create a virtual environment
sudo apt-get install python3-pip <br>
sudo pip3 install virtualenv <br>
virtualenv venv <br>
source venv/bin/activate <br>
## Install the necessary packages
pip install -r requirements.txt <br>
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib

## Run the application
python prompt.py

## Open the application from the url
