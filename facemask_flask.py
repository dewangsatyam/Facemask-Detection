
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.metrics import f1_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import shutil
import cv2
import glob

# Any results you write to the current directory are saved as output.

from pathlib import Path
from fastai import *
from fastai.vision import *
import torch
from fastai.callbacks.hooks import *
from flask import Flask, jsonify, make_response, request

app = Flask(__name__)

print("Loading the model")
learn = load_learner(r'C:\Users\Dewang\Desktop\facemask/','export.pkl')
print("Model Loaded")

@app.route('/fmclassifier', methods = ['POST'])
def predict():
    path = request.get_json()['text']
    print(path)
    img = open_image(get_image_files(path)[0])
    pred_class,pred_idx,outputs = learn.predict(img)
    pred_class = str(pred_class)
    confidence = float(max(outputs))
    return jsonify({'prediction' : pred_class, 'confidence_score' : confidence})

if __name__ == "__main__":
    app.debug = True
    app.run(host = '0.0.0.0', port = '5000')