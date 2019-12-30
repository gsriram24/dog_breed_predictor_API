import numpy as np
from glob import glob
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from tensorflow.keras.models import load_model
import os
import flask
from flask import request, jsonify
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

app = flask.Flask(__name__)

with open('dog_names.txt', 'r') as f:
    dog_names = f.readlines()


def extract_VGG19(tensor):

    return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def VGG19_predict_breed(img_path):
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]


VGG19_model = load_model('saved_models/VGG19_model.h5')


@app.route('/', methods=['POST'])
def api_id():
    file = request.files['file']
    filename = file.filename
    file_directory = os.path.join('images/', filename)
    file.save(file_directory)
    results = {}
    results['data'] = VGG19_predict_breed(file_directory)
    os.remove(file_directory)
    return jsonify(results)


if __name__ == '__main__':
    app.run(threaded=False, port=80)
