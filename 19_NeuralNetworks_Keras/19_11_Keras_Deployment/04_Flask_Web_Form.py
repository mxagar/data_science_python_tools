from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

def return_prediction(model,scaler,sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']    
    #flower = [[s_len,s_wid,p_len,p_wid]]
    flower = pd.DataFrame([[s_len,s_wid,p_len,p_wid]])
    flower.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    flower = scaler.transform(flower)
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    class_ind = model.predict_classes(flower)
    return classes[class_ind][0]

app = Flask(__name__)
# Configure a secret SECRET_KEY with an arbitrary string for now
# Necessary to allow the user to interact with the form (?)
app.config['SECRET_KEY'] = 'mysecretkey'

# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

# Create a Flask-WTForm Class inhereited from a FlaskForm
# Fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    sep_len = TextField('Sepal Length')
    sep_wid = TextField('Sepal Width')
    pet_len = TextField('Petal Length')
    pet_wid = TextField('Petal Width')
    submit = SubmitField('Analyze')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form
    form = FlowerForm()
    # If the form is valid on submission
    if form.validate_on_submit():
        # Grab the data from the breed on the form.
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data
        # Link to /prediction page below
        return redirect(url_for("prediction"))
    # If not valid/submitted, show home.html template page
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    # Create empty dictionary to fill in; that dict is the sample to feed
    content = {}
    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])
    # Run model
    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)
    # Display output on prediction.html template page
    return render_template('prediction.html',results=results)

if __name__ == '__main__':
    app.run(debug=True)

# IMPORTANT NOTE:
# We need to have a folder ./templates which contains the template HTML files referenced:
# - ./templates/home.html
# - ./templates/prediction.html
    
# How to run it:
# - python 04_Flask_Web_Form.py
# - notice the ouput URL: http://127.0.0.1:5000 or http://localhost:5000
