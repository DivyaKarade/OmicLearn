FROM amd64/python:3.7
EXPOSE 8501

COPY . .
RUN pip install -r requirements.txt
RUN pip install xgboost

CMD streamlit run ProtoLearn.py --browser.gatherUsageStats False
