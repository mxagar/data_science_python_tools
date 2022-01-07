from flask import Flask
# We create a Flask application instance
app = Flask(__name__)

# Flask works by routing HTML content from the .py to a web page
# Inside route() we write the home page we'd like to route HTML content to;
# Default is '/'
# Decorators are used:
# we wrap with invisible Flask deployment code the HTML content
@app.route('/')
def index():
    # The HTML content
    return '<h1>FLASK APP IS RUNNING!</h1>'

if __name__ == '__main__':
    app.run()
    
# How to run it:
# - conda install -c anaconda flask
# - python 01_Basic_Flask_App.py
# - open browser on URL:port in ouput: http://127.0.0.1:5000