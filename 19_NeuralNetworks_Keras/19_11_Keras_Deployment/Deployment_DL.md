# Deployment of Deep Learning Models

This section explains how the deployment of a TF/Keras model can be done with a local web application written with Flask/Python.

The example is quite rudimentary, but it fulfills its goal.

List of files in it (in order of creation & relevance):

- `19_11_1_Keras_Deployment_Model_Iris.ipynb`: the model is defined and trained for the Iris dataset. Then, the code to be deployed is collected in a cell, ready for copy & paste. After creating the model, two files are saved:
    - `final_iris_model.h5`: the final trained TF/Keras model
    - `iris_scaler.pkl`: the scaler used to transform the data
- `01_Basic_Flask_App.py`: very simple Flask app in which a web application with a greeting is created.
- `02_Flask_API.py`