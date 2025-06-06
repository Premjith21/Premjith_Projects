import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json  # directly get JSON object
    if data is None:
        return jsonify({"error": "No JSON data found"}), 400

    print("Received data:", data)

    try:
        # Convert data values to numpy array for scaling and prediction
        features = np.array(list(data.values())).reshape(1, -1)
        new_data = scalar.transform(features)
        output = regmodel.predict(new_data)
        print("Prediction:", output[0])
        return jsonify(output[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     