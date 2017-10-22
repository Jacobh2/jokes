FROM tensorflow/tensorflow:1.3.0-py3

MAINTAINER Jacob Hagstedt <jacob.hagstedt@gmail.com>

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install --upgrade --quiet -r requirements.txt

# Install punkt tokenizer
RUN python -m nltk.downloader punkt

# COPY phantomjs-2.1.1-linux-x86_64.tar.bz2 phantomjs-2.1.1-linux-x86_64.tar.bz2
RUN wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2

RUN tar -xvf phantomjs-2.1.1-linux-x86_64.tar.bz2 && mv /usr/app/src/phantomjs-2.1.1-linux-x86_64/bin/phantomjs /usr/app/src/phantomjs && rm -r /usr/app/src/phantomjs-2.1.1-linux-x86_64

RUN apt-get update && apt-get install -y libfontconfig

COPY . .

EXPOSE 8080

# Command to run when starting the service
CMD [ "uwsgi", "--ini", "uwsgi.ini"]