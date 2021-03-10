# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:55:09 2021

@author: HP
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the Model
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        y_probabilities_test = model.predict_proba(final_features)
        y_prob_success = y_probabilities_test[:, 1]
        print("final features",final_features)
        print("prediction:",prediction)
        output = round(prediction[0], 2)
        y_prob=round(y_prob_success[0], 3)
        print(output)
    

      

        if output == 0:
            return render_template('index.html', prediction_text='Danger! you have to take care of yourself. Tumor is Malignant{}'.format(y_prob))
        else:
            return render_template('index.html', prediction_text=' Great, you need not to worry. Tumor is Benign {}'.format(y_prob))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

