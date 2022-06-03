import pickle
from math import log10
import perceptor
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

app  = Flask(__name__)

@app.route('/')
#@app.route('/index')
def home():
    return "<h1>Strona poczatkowa</h1>"

@app.route('/hello/<name>')
def success(name):
    return f'<h2>{name}</h2>'



# Create an API end point
@app.route('/api/v1.0/predict', methods=["GET"])
def get_prediction():

    # sepal length
    sepal_length = float(request.args.get('sl'))
    # sepal width
    #sepal_width = float(request.args.get('sw'))
    # petal length
    petal_length = float(request.args.get('pl'))
    # petal width
    #petal_width = float(request.args.get('pw'))

    # The features of the observation to predict
    #features = [sepal_length,
    #            sepal_width,
    #            petal_length,
    #           petal_width]
    
    features = [sepal_length,
                petal_length]
    
    print(features)
    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
    print(model)
    # Predict the class using the model
    predicted_class = int(model.predict(features))
    
    # Return a json object containing the features and prediction
    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(host = "0.0.0.0")
