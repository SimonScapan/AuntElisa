# coding=utf8
from flask import Flask, request
from flask_cors import CORS
from auntelisa import Therapist

APP = Flask(__name__, static_folder="build/static", template_folder="build")
CORS = CORS(APP)

@APP.route('/backend/<text>', methods=["GET"])
def chatbot(text):
    foo = Therapist(text)
    return foo

if __name__ == '__main__':
    APP.run(debug=True, host='0.0.0.0')