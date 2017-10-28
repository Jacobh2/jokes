FROM resin/rpi-raspbian:jessie

# First, install the dependencies for TensorFlow:
RUN apt-get update && apt-get install -y wget python3-pip python3-dev build-essential

#Next, download the wheel file from this repository...
RUN wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl

# ... and install it:
RUN pip3 install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl

# Finally, we need to reinstall the mock library to keep it from throwing an error when we import TensorFlow:
RUN pip3 uninstall mock
RUN pip3 install mock

# Cleanup to decrease image size
RUN apt-get autoremove