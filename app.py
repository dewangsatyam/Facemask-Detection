from flask import Flask
from flask_restful import Resource, Api, reqparse
import os
from werkzeug.datastructures import FileStorage
import random
from fastai import *
from fastai.vision import *
import torch
from fastai.callbacks.hooks import *

app = Flask(__name__)
api = Api(app)

print("Loading the model")
learn = load_learner('./','export.pkl')
print("Model Loaded")

@app.route('/')
def index():
    return "Face Mask Detection"

class UploadImage(Resource):
    def post(self):
        # Route to receive image files
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        img = args['image']
        file_name = random_string()
        img.save(f"./static/uploaded_images/{file_name}.jpeg")
        test_img = open_image(f"./static/uploaded_images/{file_name}.jpeg")
        pred_class, pred_idx, outputs = learn.predict(test_img)
        pred_class = str(pred_class)
        confidence = float(max(outputs))
        os.remove(f'./static/uploaded_images/{file_name}.jpeg')
        return {'prediction' : pred_class, 'confidence_score' : confidence}

def random_string():
  return ''.join([chr(random.randint(97,122)) for i in range(10)])

if __name__ == "app":
    path = './static/uploaded_images'
    if not os.path.exists(path):
        os.makedirs(path)
    api.add_resource(UploadImage, '/check')