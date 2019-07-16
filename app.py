from flask import Flask, flash, jsonify, redirect, render_template, request, session
import NN as nn
import re
import base64
from scipy.misc import imsave, imread, imresize
import numpy as np

# Configure application
app = Flask(__name__)

# Load trained network
net = nn.load_network("BestNetwork.json")

# Reload templates when they are changed
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.after_request
def after_request(response):
    """Disable caching"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    if request.method == "POST":
        parseImage(request.get_data())

        # read parsed image back in 8-bit, black and white mode (L)
        x = imread('output.png', mode='L')
        x = np.invert(x)
        x = imresize(x,(28,28))

        # reshape image data for use in neural network
        x = x.reshape(784, 1) / 255.0

        # Predict the nuber drawn
        guess = net.feedforward(x)

        prediction = np.array2string(np.argmax(guess))
        print(prediction)
        
        return prediction
    return "Internal server error"


def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))