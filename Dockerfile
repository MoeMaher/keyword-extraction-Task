FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=5000 -r requirements.txt

COPY . /usr/src/app



