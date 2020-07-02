# coding=utf8
from flask import Flask, request
from flask_cors import CORS
from auntelisa import Therapist

# initialize Flask
APP = Flask(__name__, static_folder="build/static", template_folder="build")
CORS = CORS(APP)

# listen to GET method and compute input text with chatbot and give back to frontend
@APP.route('/backend/<text>', methods=["GET"])
def chatbot(text):
    foo = Therapist(text)
    return foo

# run APP on localhost
if __name__ == '__main__':
    APP.run(debug=True, host='0.0.0.0')