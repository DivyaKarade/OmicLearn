FROM python:3.9-slim-buster

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501