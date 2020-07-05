# proto_learn
Machine learning for proteomics data

## Installation

* `conda create --name proto-learn python=3.7`
* `pip install -r requirements.txt`

To use Xgboost:
* `conda install py-xgboost`

To use SVVG export:

* `conda install selenium geckodriver firefox -c conda-forge`

## Running

`streamlit run proto_learn.py --browser.gatherUsageStats False`


## Todo

* Check selenium integration and deactivate if not installed. 
