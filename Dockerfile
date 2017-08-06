FROM tensorflow/tensorflow:1.2.1-py3

MAINTAINER Jacob Hagstedt <jacob.hagstedt@gmail.com>

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install --upgrade --quiet -r requirements.txt

RUN python -m nltk.downloader punkt

COPY . .

CMD ["python", "-m", "preprocessing.process"]