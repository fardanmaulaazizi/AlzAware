import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import os
import numpy as np
from keras.preprocessing import image
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
from PIL import Image

app = Flask(__name__)

# load model for prediction
model = load_model("modelAlzheimer.h5")

# Variabel
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
temp_filename = "temp_image.png"
img_url = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

def allowed_file(temp_filename):
    return '.' in temp_filename and temp_filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("home.html")

@app.route("/classification", methods = ['GET', 'POST'])
def classification():
    if 'file' not in request.files:
	    return render_template("classify.html")

    files = request.files.getlist('file')
    errors = {}

    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)
    
    # convert image to RGB
    img = Image.open(img_url)
    now = datetime.now()
    predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    image_predict = predict_image_path
    img.save(image_predict, format="png")
    img.close()

    # prepare image for prediction
    img = image.load_img(predict_image_path, target_size=(128, 128, 3))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Predict Image
    prediction_array = model.predict(images)
    result = {
        "img_path" : predict_image_path,
        "prediction": class_names[np.argmax(prediction_array)],
        "confidence": '{:2.0f}%'.format(100 * np.max(prediction_array))
    }

    return render_template("classify.html", img_path = result["img_path"], prediction = result["prediction"],confidence = result["confidence"], errors = errors)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)