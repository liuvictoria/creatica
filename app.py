import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json

# Some utilites
import numpy as np
from util import base64_to_pil

# Pre-process image with inceptionnet
import preprocess_one_hotdog


TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('static')

# app = Flask(__name__) # to make the app run without any
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


# Declare a flask app
app = Flask(__name__)



# Remove src from cwd if necessary
cwd = os.getcwd()
if os.path.basename(cwd) == 'src': cwd = os.path.dirname(cwd)

model_name = 'multiclass_debugging'
    
# Create img directory to save images if needed
os.makedirs(os.path.join(cwd, 'img'), exist_ok=True)

# Create model directory to save models if needed
os.makedirs(os.path.join(cwd, 'model'), exist_ok=True)
model_weights_fname = os.path.join(cwd, 'model', model_name + '.h5')
model_json_fname = os.path.join(cwd, 'model', model_name + '.json')

# Load model and its weights
with open(model_json_fname, 'r') as f: model_json = f.read()
model = model_from_json(model_json)
model.load_weights(model_weights_fname)

# Compile model and evaluate its performance on training and test data
model.compile(loss='binary_crossentropy', optimizer='adam',
    metrics=['accuracy'])

#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    rounded_predictions = np.argmax(model.predict(img, verbose=1), axis = -1)
    print(rounded_predictions)
    if rounded_predictions[0] == 0:
        prediction = 'g0bAnaNAs'
    elif rounded_predictions[0] == 1:
        prediction = '^carrot^'
    else:
        prediction = 'h0tt0d0gg0'
    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save(os.path.join(cwd, "demo/test/1.png"))
        print(os.path.join(cwd, "demo/test/1.png"))
        
        #preprocess image with inceptionNetV3
        #use script preprocess_one_hotdog, which should be in same directory
        img = preprocess_one_hotdog.get_single_image_data()
        
        # Make prediction
        preds = model_predict(img, model)

        
        # Serialize the result, you can add additional fields
        return jsonify(
            result=preds,
        )

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
