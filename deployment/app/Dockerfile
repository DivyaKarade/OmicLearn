FROM python:3.8-slim-buster AS compile-image

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./deployment/app/requirements.txt /app/

RUN pip install --upgrade pip

RUN pip install -r app/requirements.txt

FROM python:3.8-slim-buster

COPY --from=compile-image /opt/venv /opt/venv

WORKDIR /app

COPY . /app/

EXPOSE 8501

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/opt/venv/bin:$PATH"