## **Table of Contents**
- [Installation](#installation)
- [Running](#running)
- [Running with Docker](#running-with-docker)

## Installation

> To be able to install Proto Learn and run it, you may prefer to install Anaconda if you did not already. 
> 
> For more details, please visit the [Installation Page of Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

- Here are the commands for creating an environment and install all of the requirements at once:

`conda create --name proto-learn python=3.7`

`pip install -r requirements.txt`

- To use Xgboost:

`conda install py-xgboost`

- To use SVG export:

`conda install selenium geckodriver firefox -c conda-forge`

- To use Bokeh plot without any problem:

`conda install -c conda-forge phantomjs`

## Running

- Now, time to run Proto Learn!

`streamlit run proto_learn.py --browser.gatherUsageStats False`

## Running with Docker
> Need to rely on the relatively large conda environment

`docker build -f Dockerfile -t proto_learn:latest .`

`docker run -p 8501:8501 proto_learn:latest`

- Then, you can access Proto Learn from `http://localhost:8501`