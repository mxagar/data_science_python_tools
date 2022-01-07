# Deployment of Deep Learning Models

This section/folder explains how the deployment of a TF/Keras model can be done with a local web application written with Flask/Python.

I created the section/folder while coding along the Udemy course by J.M. Portilla

[Complete Tensorflow 2 and Keras Deep Learning Bootcamp](https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/)

The examples are quite rudimentary, but it fulfills its goal.
However, I am not sure the steps followed here are a best practice in terms of CI/CD...

Basically, the following things are accomplished:
- A model is created on a notebook and exported to a file, together with the data scaler.
- A Flask-based web-app is created which offers a custom API; that API receives a JSON of a new sample and returns the prediction.
- We use Postman for testing the API.
- We use then python to test the API.
- A web form is created with Flask-WTF so that the user can query the class of a sample with a GUI.
- Finally, the local web application is deployed to the internet using `heroku`, which is free.

List of files in it (in order of creation & relevance):

- `19_11_1_Keras_Deployment_Model_Iris.ipynb`: the model is defined and trained for the Iris dataset. Then, the code to be deployed is collected in a cell, ready for copy & paste. After creating the model, two files are saved:
    - `final_iris_model.h5`: the final trained TF/Keras model
    - `iris_scaler.pkl`: the scaler used to transform the data
- `01_Basic_Flask_App.py`: very simple Flask app in which a web application with a greeting is created.
- `02_Flask_API.py`: the previous python file is updated to contain the deployment code from the notebook. As a result, we create a Flask app that has an API which can be accessed to request predictions upon new samples. We can use the API with `curl`, `python-requets`, etc. A very common and simple option for testing APIs is Postman. See next section for more details on how to perform an API request using Postman.
- `03_Python_Request.py`: python script that performs a request to the API started with the previous file; in other words, this program sends a JSON with a sample to the web-app with model and receives the class of flower from it.
- `04_Flask_Web_Form.py`: 

## Postman API Requests & Python Requests

Postman is a tool for building and using APIs.
We can very easily use Postman to test an API, ours or from 3rd parties.
It is possible to do it online and also locally, with the `Postman.app`.

First, we need to start our Flask-based web API:

```bash
python 02_Flask_API.py
# Notice the output URL: http://127.0.0.1:5000 or http://localhost:5000
```

Then, we open `Postman.app` and

- Create New (HTTP) Request; Save, e.g., as 'Flower Request' in Collection 'Keras Development Tests'
- Type of request: POST
- URL: http://127.0.0.1:5000/api/flower
- Select Body, raw
- Select JSON (instead of Text)
- Type or Paste in example JSON request

```json
{"sepal_length":5.4,
 "sepal_width":3.3,
 "petal_length":1.2,
 "petal_width":0.15}
```

- Press 'Send'
- We should get the response: "setosa"

Note that in `Postmapp.app` we can get the equivalent request codes to our GUI request, such as `curl` or `python-request`!
Just click on the code button of the right vertical menu :)

For instance, we see that the **Python - Requests** equivalent code is:

```python
import requests
import json

url = "http://127.0.0.1:5000/api/flower"

payload = json.dumps({
  "sepal_length": 5.4,
  "sepal_width": 3.3,
  "petal_length": 1.2,
  "petal_width": 0.15
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

# That should return 200 if correct
#print(response.status_code)

print(response.text)
```

## Flask Web Form

To install Flask and Flask Web-Form Tools:

```bash
conda install -c anaconda flask
conda install -c anaconda flask-wtf
```

