FROM continuumio/miniconda3
EXPOSE 8501

RUN conda create -n proto-learn python=3.7
RUN /bin/bash -c "source activate proto-learn"
COPY . .
RUN pip install -r requirements.txt
RUN conda install py-xgboost

CMD streamlit run proto_learn.py --browser.gatherUsageStats False
