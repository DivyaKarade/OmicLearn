FROM continuumio/miniconda3
EXPOSE 8501

RUN conda create -n env python=3.7
RUN /bin/bash -c "source activate env"

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
RUN conda install py-xgboost
RUN conda install selenium geckodriver firefox -c conda-forge

COPY . .

CMD streamlit run proto_learn.py --browser.gatherUsageStats False
