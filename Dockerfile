FROM continuumio/miniconda3
EXPOSE 8501

RUN conda create -n proto-learn python=3.7
RUN /bin/bash -c "source activate proto-learn"

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
RUN conda install py-xgboost

COPY . .

CMD streamlit run ProtoLearn.py --browser.gatherUsageStats False
