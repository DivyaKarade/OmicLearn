## **Table of Contents**
- [Installation instructions](#installation-instructions)
- [Running](#running)
- [Running with Docker](#running-with-docker)

---

## Installation instructions

> We highly recommend the [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distribution which comes with a powerful package manager. 
>
> It is strongly recommended to install OmicLearn in its own environment.

1. Open the console and create a new conda environment: `conda create --name omic_learn python=3.7`
2. Activate the environment: `conda activate omic_learn` for Linux / Mac Os X / Windows

>
> **Note:** Type following command for conda versions prior to `4.6`:
>
> `source activate omic_learn` for macOS and Linux
>
> `activate omic_learn` for Windows


3. Redirect to the folder of choice and clone the repository: `git clone https://github.com/OmicEra/OmicLearn`
4. Install the packages with `pip install -r requirements.txt`
5. To be able to use Xgboost, install via conda: `conda install py-xgboost`

## Running

- To run OmicLearn, type the following command:

`streamlit run omic_learn.py --browser.gatherUsageStats False`

[Streamlit](https://www.streamlit.io/) gathers usage statistics per default, which we disable with this command.

> **Note:** A vanilla streamlit installation will show a menu bar in the upper left corner that offers additional functionality, such as recording screencasts. 
>
> For OmicLearn, this functionality is disabled. 

## Running with Docker

A docker instance should have at least 4 GB of memory. 
To build the docker, navigate to the OmicLearn directory: 

* `docker build -f Dockerfile -t omic_learn:latest .`

To run the docker container type:
* `docker run -p 8501:8501 omic_learn:latest`

* The OmicLearn page will be accessible via [`http://localhost:8501`](http://localhost:8501)