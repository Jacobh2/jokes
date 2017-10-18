FROM elswork/rpi-tensorflow:1.3.0V1

MAINTAINER Jacob Hagstedt <jacob.hagstedt@gmail.com>

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install --upgrade --quiet -r requirements.txt

COPY . .

EXPOSE 8080

# Command to run when starting the service
CMD [ "uwsgi", "--ini", "uwsgi.ini"]