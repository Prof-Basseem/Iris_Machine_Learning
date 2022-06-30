# deploment machine learning iris dataset by pki file using flask framework
# import flask framework
from flask import Flask, render_template, request, redirect, url_for
# import pandas library
import pandas as pd
# import numpy library
import numpy as np
# pikel library for machine learning
import pickle

# create flask object
app = Flask(__name__)

# load pkl file
model = pickle.load(open('model.pkl', 'rb'))

# route to index.html
@app.route('/')
def home():
    return render_template('index.html')

# route to prediction.html page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width']) 
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        # make a prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        # return prediction
        if prediction[0] == 0:
            output = 'Iris-setosa'
        elif prediction[0] == 1:
            output = 'Iris-versicolor'
        else:
            output = 'Iris-virginica'
        return render_template('predict.html', prediction_text='The flower is {}'.format(output))
    else:
        return render_template('index.html')
# run the app
if __name__ == "__main__":
    app.run(debug=True)
# end of file

