from flask import Flask, render_template, request
import joblib

# Load the saved model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Replace with your HTML template name

import numpy as np

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        data = request.form.to_dict()  # Gets form data as a dictionary

        # Convert input data to numerical values
        inputs = np.array([float(data["input_1"]), float(data["input_2"]), float(data["input_3"]), float(data["input_4"]), float(data["input_5"]), float(data["input_6"])]).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(inputs)

        return render_template("result.html", prediction=prediction[0])  # Replace with result template
    else:
        return render_template("result.html")  # Display form for GET request

if __name__ == "__main__":
    app.run(debug=True)
