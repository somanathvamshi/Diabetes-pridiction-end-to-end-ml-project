import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diabetes_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output==1:
        prediction_text = 'the person has diabetes'
    else:
        prediction_text = 'the person has no diabetes'

    return render_template('index.html', prediction_text=prediction_text)
if __name__ == "__main__":
    app.run(debug=True)