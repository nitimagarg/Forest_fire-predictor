import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app_logger import log
from dotenv import load_dotenv
load_dotenv()
from mongodb import mongodbconnection
import warnings
import os
warnings.filterwarnings("ignore")
envs= load_dotenv()
print(envs)
app = Flask(__name__)
username = os.getenv('MONGO_USERNAME')
password = os.getenv('MONGO_PASSWORD')
print(username,password)
# Import Classification and Regression model file
pickle_in = bz2.BZ2File('model/classification.pkl', 'rb')
R_pickle_in = bz2.BZ2File('model/regression.pkl', 'rb')
model_C = pickle.load(pickle_in)
model_R = pickle.load(R_pickle_in)

# Data retrieved from DB using mongoconnection module
dbcon = mongodbconnection(username=username,password=password)
list_cursor = dbcon.getdata(dbName='FireDataML', collectionName='ml_task')
log.info('Connected to Mongodb and data retrieved')

# Data From MongoDB is used for Standardization
df = pd.DataFrame(list_cursor)
df.drop('_id', axis=1, inplace=True)
log.info('DataFrame created')
scaler = StandardScaler()
X = df.drop(['FWI', 'Classes'], axis=1)
# Standardize
X_reg_scaled = scaler.fit_transform(X)
log.info('Standardization done')


# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')


# Route for API Testing
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        json_data = request.get_json()
        data = json_data.get('data')
        if data is None:
            return jsonify({"error": "'data' field missing in request"}), 400
        log.info('Input from Api testing', data)
        new_data = [list(data.values())]
        final_data = scaler.transform(new_data)
        output = int(model_C.predict(final_data)[0])
        if output == 1:
            text = 'The Forest in Danger'
        else:
            text = 'Forest is Safe'
        return jsonify(text, output)
    except Exception as e:
        output = 'Check the in input again!'
        log.error('error in input from Postman', e)
        return jsonify(output)


# Route for Classification Model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_features = [np.array(data)]
        final_features = scaler.transform(final_features)
        output = model_C.predict(final_features)[0]
        log.info('Prediction done for Classification model')
        if output == 0:
            text = 'Forest is Safe!'
            result_html = f'<span class="prediction-safe"><b>{text} --- Chance of Fire is {output}</b></span>'
        else:
            text = 'Forest is in Danger!'
            result_html = f'<span class="prediction-danger"><b>{text} --- Chance of Fire is {output}</b></span>'
        return render_template('index.html', prediction_text1=result_html)
    except Exception as e:
        log.error('Input error, check input', e)
        return render_template('index.html', prediction_text1="Check the Input again!!!")


# Route for Regression Model
@app.route('/predictR', methods=['POST'])
def predictR():
    try:
        data = [float(x) for x in request.form.values()]
        data = [np.array(data)]
        data = scaler.transform(data)
        output = model_R.predict(data)[0]
        log.info('Prediction done for Regression model')
        if output > 15:
            result_html = f'<span class="prediction-danger"><b>Fuel Moisture Code index is {output:.4f} ---- Warning!!! High hazard rating</b></span>'
        else:
            result_html = f'<span class="prediction-safe"><b>Fuel Moisture Code index is {output:.4f} ---- Safe.. Low hazard rating</b></span>'
        return render_template('index.html', prediction_text2=result_html)
    except Exception as e:
        log.error('Input error, check input', e)
        return render_template('index.html', prediction_text2="Check the Input again!!!")


# Run APP in Debug mode
if __name__ == "__main__":
    app.run(debug=True)