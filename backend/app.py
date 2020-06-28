# coding=utf8
from flask import Flask, request
from flask_cors import CORS
import aunt_elisa
import json

APP = Flask(__name__, static_folder="build/static", template_folder="build")
CORS = CORS(APP)


# GET - output from Therapist
# SET - input data to Therapist

@APP.route('/backend', methods=["GET", "SET"])
def bot():
    if request.method == "GET":
        response = json.dumps(Therapist(text))
    elif request.method == "SET":
        text = json.dumps(request.json)
    return response

if __name__ == '__main__':
    APP.run(debug=True, host='0.0.0.0')