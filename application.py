import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from tensorflow.keras.models import load_model
from extract_bottleneck_features import *
import os
import flask
from flask import request, jsonify
from tensorflow.keras.applications.resnet50 import ResNet50


app = flask.Flask(__name__)

with open('dog_names.txt', 'r') as f:
    dog_names = f.readlines()

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


ResNet50_model = ResNet50(weights='imagenet')


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def VGG19_predict_breed(img_path):
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]


VGG19_model = load_model('VGG19_model.h5')
VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')


@app.route('/', methods=['POST'])
def api_id():
    file = request.files['file']
    filename = file.filename
    file_directory = os.path.join('images/', filename)
    file.save(file_directory)
    results = {}
    if(dog_detector(file_directory)):
        results['type'] = 'dog'
    elif(face_detector(file_directory)):
        results['type'] = 'human'
    else:
        results['type'] = 'unknown'
    results['data'] = VGG19_predict_breed(file_directory)
    os.remove(file_directory)
    return jsonify(results)


app.run(threaded=False)
