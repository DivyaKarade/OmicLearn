<p align="center"> <img src="https://user-images.githubusercontent.com/49681382/101802266-48204a00-3b20-11eb-85ec-08c123fca79e.png" height="270" width="277" /> </p>
<h2 align="center"> 📰 Manual and Documentation is available at: <a href="https://github.com/OmicEra/OmicLearn/wiki" target="_blank">OmicLearn Wiki Page </a> </h2>
<h2 align="center"> 🟢 OmicLearn is now accessible here: <a href="http://omiclearn.com/" target="_blank">omiclearn.com</a> </h2>

![OmicLearn Tests](https://github.com/OmicEra/OmicLearn/workflows/OmicLearn%20Tests/badge.svg)
![OmicLearn Python Badges](https://img.shields.io/badge/Tested_with_Python-3.7-blue)
![OmicLearn Version](https://img.shields.io/badge/Release-v1.0.0-orange)
![OmicLearn Release](https://img.shields.io/badge/Release%20Date-February%202021-green)
![OmicLearn License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

---

## OmicLearn

Transparent exploration of machine learning approaches for omics datasets.

## Installation & Running

> More information about `Installation & Running` is available on our **[Wiki pages](https://github.com/OmicEra/OmicLearn/wiki/HOW-TO:-Installation-&-Running)**.

- It is strongly recommended to install OmicLearn in its own environment using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

  1. Open the console and create a new conda environment: `conda create --name omic_learn python=3.7`
  2. Activate the environment: `conda activate omic_learn` for Linux / Mac Os X / Windows
  
  
  > **Note:** Type following command for conda versions prior to `4.6`:
  >
  > `source activate omic_learn` for macOS and Linux
  >
  > `activate omic_learn` for Windows

  3. Redirect to the folder of choice and clone the repository: `git clone https://github.com/OmicEra/OmicLearn`
  4. Install the required packages with `pip install -r requirements.txt`
  5. To be able to use Xgboost, install via conda: `conda install py-xgboost`

- After successfull instalattion, type the following command to run OmicLearn:

  `streamlit run omic_learn.py --browser.gatherUsageStats False`
  
  > `Running with Docker` option is also available. Please, check the installation instructions on **[the Wiki pages](https://github.com/OmicEra/OmicLearn/wiki/INSTALLATION-%26-RUNNING/)**.
  
 - After starting the streamlit server, the OmicLearn page should be automatically opened in your browser (Default link: [`http://localhost:8501`](http://localhost:8501) 

## Getting Started with OmicLearn

The following image displays the main steps of OmicLearn:

![OmicLearn Workflow](https://user-images.githubusercontent.com/49681382/91734594-cb421380-ebb3-11ea-91fa-8acc8826ae7b.png)

Detailed instructions on how to get started with OmicLearn can be found **[here.]**(https://github.com/OmicEra/OmicLearn/wiki/HOW-TO:-Using)

On this page, you can click on the titles listed in *Table of Contents*, which contains instructions for eachs ection.

## Contributing
All contributions are welcome. 👍

📰 To get started, please check out our **[`CONTRIBUTING`](https://github.com/OmicEra/OmicLearn/blob/master/CONTRIBUTING.md)** guidelines. 

When contributing to **OmicLearn**, please **[open a new issue](https://github.com/OmicEra/OmicLearn/issues/new/choose)** to report the bug or discuss the changes you plan before sending a PR (pull request).

Also, be aware that you agree to the **`OmicEra Individual Contributor License Agreement`** by submitting your code. 🤝
