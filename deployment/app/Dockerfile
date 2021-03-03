FROM amd64/python:3.7
EXPOSE 8501

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD streamlit run omic_learn.py --browser.gatherUsageStats False
