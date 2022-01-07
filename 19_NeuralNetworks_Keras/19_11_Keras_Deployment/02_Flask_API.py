from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib
# Watch out: new modules loaded: request, jsonify
from flask import Flask, request, jsonify

# Our prediction function
def return_prediction(model,scaler,sample_json):
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]
    #flower = [[s_len,s_wid,p_len,p_wid]]
    flower = pd.DataFrame([[s_len,s_wid,p_len,p_wid]])
    flower.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    flower = scaler.transform(flower)
    class_ind = model.predict_classes(flower)[0]
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[class_ind]

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'

# We need to load our models/objects used in the prediction function
flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

# We create a new routing page
@app.route('/api/flower',methods=['POST'])
def flower_prediction():
    content = request.json
    results = return_prediction(flower_model,flower_scaler,content)
    return jsonify(results)
    
if __name__ == '__main__':
    app.run()
    
# How to run it:
# - conda install -c anaconda flask
# - python 02_Flask_API.py
# - notice the ouput URL: http://127.0.0.1:5000 or http://localhost:5000
# - open Postman.app
# -- Create New (HTTP) Request; Save, e.g., as 'Flower Request' in Collection 'Keras Development Tests'
# -- Type of request: POST
# -- URL: http://127.0.0.1:5000/api/flower
# -- Select Body, raw
# -- Select JSON (instead of Text)
# -- Type or Paste in example JSON request
#      {"sepal_length":5.4,
#       "sepal_width":3.3,
#       "petal_length":1.2,
#       "petal_width":0.15}
# -- Press 'Send'
# -- We should get the response: "setosa"