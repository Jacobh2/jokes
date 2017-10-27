FROM elswork/rpi-tensorflow:1.3.0V1

MAINTAINER Jacob Hagstedt <jacob.hagstedt@gmail.com>

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install --upgrade --quiet -r requirements.txt

# Install punkt tokenizer
RUN python -m nltk.downloader punkt

COPY . .

EXPOSE 8080

RUN alias python=python3

# Command to run when starting the service
CMD [ "uwsgi", "--ini", "uwsgi.ini"]
