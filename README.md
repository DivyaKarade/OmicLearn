# proto_learn
Machine learning for proteomics data

## Installation

* `conda create --name proto-learn python=3.7`
* `pip install -r requirements.txt`

To use Xgboost:
* `conda install py-xgboost`

To use SVG export:

* `conda install selenium geckodriver firefox -c conda-forge`

## Running

`streamlit run proto_learn.py --browser.gatherUsageStats False`

## Running Docker
Need to rely on the relatively large conda environment

`docker build -f Dockerfile -t proto_learn:latest .`
`docker run -p 8501:8501 proto_learn:latest`

Access via `http://localhost:8501`

## Todo

* Check selenium integration and deactivate if not installed.
