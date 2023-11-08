#!pip install Flask

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as model:
    iris = pickle.load(model)
# iris = pickle.load("model.pkl", "rb")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare the input data for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make predictions
        prediction = iris.predict(input_data)

        return render_template('/prediction.html', prediction=prediction[0])
    return render_template('/index.html')

if __name__ == '__main__':
    app.run(debug=True)
