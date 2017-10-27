FROM elswork/rpi-tensorflow:1.3.0V1

MAINTAINER Jacob Hagstedt <jacob.hagstedt@gmail.com>

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --upgrade --quiet -r requirements.txt

# Install punkt tokenizer
RUN python3 -m nltk.downloader punkt

COPY . .

EXPOSE 8080

RUN echo "alias python=python3" > ~/.bash_aliases

# Command to run when starting the service
CMD [ "uwsgi", "--ini", "uwsgi.ini"]
