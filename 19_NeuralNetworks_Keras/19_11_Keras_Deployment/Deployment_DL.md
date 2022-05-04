# Deployment of Deep Learning Models

This section/folder explains how the deployment of a TF/Keras model can be done with a local web application written with Flask/Python.
Note that as far as I understand the guideline is not specific to TF/Keras, but virtually any ML model could be deployed this way.

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

List of files in it (in order of creation & of learning progression):

- `19_11_1_Keras_Deployment_Model_Iris.ipynb`: the model is defined and trained for the Iris dataset. Then, the code to be deployed is collected in a cell, ready for copy & paste. After creating the model, two files are saved:
    - `final_iris_model.h5`: the final trained TF/Keras model
    - `iris_scaler.pkl`: the scaler used to transform the data
- `01_Basic_Flask_App.py`: very simple Flask app in which a web application with a greeting is created.
- `02_Flask_API.py`: the previous python file is updated to contain the deployment code from the notebook. As a result, we create a Flask app that has an API which can be accessed to request predictions upon new samples. We can use the API with `curl`, `python-requets`, etc. A very common and simple option for testing APIs is Postman. See next section for more details on how to perform an API request using Postman.
- `03_Python_Request.py`: python script that performs a request to the API started with the previous file; in other words, this program sends a JSON with a sample to the web-app with model and receives the class of flower from it.
- `04_Flask_Web_Form.py`: python script that creates a Flask-based web form with which the user interacts by introducing the values of the four features of a sample; when submit button is pressed, the model is run and the flower class is displayed. The web form can be run locally or hosted on a server on the internet (e.g., Heroku). Two HTML template files are created for it to run:
  - `templates/home.html`
  - `templates/prediction.html`
- `heroku_deployment/`: folder which contains files copied from the upper folder to perform the deployment of the model on Heroku (see last section below). 

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

The python file `04_Flask_Web_Form.py` is a vanilla example which:
- Creates the web-app that holds the DL model
- Displays a web form on a local URL which can be filled in by users: sample feature values can be introduced
- After clicking on submit/analyze, the result is displayed

To execute it:

```bash
python 04_Flask_Web_Form.py
```

## Deployment beyond Local Web Page: Heroku

There are many options to deploy a model wrapped on a Flask application on the internet: AWS, Azure, etc.; we can just google "Flask web app deploy..." for available options and tutorials. A possible free option is [Heroku](https://www.heroku.com).

### Step 1: Create the deployment folder

Copy the deployment files to the created folder:

```bash
mkdir heroku_deployment
cp 04_Flask_Web_Form.py heroku_deployment/app.py
cp final_iris_model.h5 heroku_deployment/final_iris_model.h5
cp iris_scaler.pkl heroku_deployment/iris_scaler.pkl
cp -r templates heroku_deployment/templates
```

That folder needs to be added to a heroku git repository; therefore, if we are working on a git repository, **we need to add it to .gitignore**.

### Step 2: Create a Heroku account and download the Heroku CLI

- Create a free Heroku account.
- Download & install Heroku CLI (google it); that is via `brew` in Mac. Note that we need to have git installed, too.

### Step 3: Prepare the contents of the deployment folder

- Create a new conda environment and installl **only with `pip`** the necessary packages. It needs to be with `pip` because we need to generate a `pip freeze` requirements file for installing on the Heroku servers.

```bash
cd ./heroku_deployment
# We need a an environment with this specific version of Python
conda create --name heroku_deployment_env python=3.7
conda activate heroku_deployment_env
# Install required packages only with pip
pip install flask
pip install Flask-WTF
pip install scikit-learn
pip install tensorflow
# G Unicorn allows us to push our flask app to the Heroku servers
pip install gunicorn
pip install pandas
# Create requirements file
pip freeze > requirements.txt
```

- Create a process file, `vim ./Procfile`, with the following content

```
web: gunicorn app:app
```

- Create a runtime Python version file, `vim ./runtime.txt`, with the following content

```
python-3.7.11
```

### Step 4:  Create a new Heroku app and deploy our model

Go to the [Heroku dashboard](https://dashboard.heroku.com/apps) and create a new app:
- Name, e.g.: `iris-classificator`
- Region: Europe
- Create
- Choose: Heroku Git (Deploy)
- Follow the setup instructions there: we basically create a git repository of our deplyment folder and connect it to the remote Heroku site.

```bash
# Log in: a web site should open
heroku login
# Create a git repo
cd heroku_deployment/
git init
heroku git:remote -a iris-classificator
# Add files to repo and DEPLOY
git add .
git commit -am "adding files to repo"
git push heroku master
```

- If changes done or errors during deployment need to be corrected, modify the files are re-run a propper version of the last 3 lines

**However, I could not manage to run my model on Heroku. The reason was that the slug was too big (600 MB > 500 MB)**. I did not spend much time trying to solve the issue, so I left it untested -- it seems to work for the instructor, though.

