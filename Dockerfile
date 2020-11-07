FROM amd64/python:3.7
EXPOSE 8501

COPY . .
RUN pip install -r requirements.txt
RUN pip install xgboost

CMD streamlit run omic_learn.py --browser.gatherUsageStats False
