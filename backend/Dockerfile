# Setup Linux-Envoirement
# FROM ubuntu:16.04
FROM python:3.7.5-slim

# install pip, python, libpq
RUN apt-get update -y && \
     apt-get install -y python-pip python-dev libpq-dev

# Copy the requirements to the containter and change WORKDIR
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app

# Install all python-libaries from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY ./ /app

# Set the startingpoint for the container
ENTRYPOINT [ "python" ]
CMD ["app.py" ] 