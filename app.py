from flask import Flask, render_template, request
import pandas as pd
import torch.nn as nn
import torch
import os
from dotenv import load_dotenv
import requests
import json
from joblib import load

app = Flask(__name__)

# Load the .env file and access your API key
load_dotenv()
api_key = os.getenv('API_KEY')

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Load the model and its parameters
model = LinearRegressionModel(81)
model.load_state_dict(torch.load('linear_regression_model.pth'))
column_transformer = load('column_transformer.joblib')

def transform_data(df, column_transformer):
    df = pd.json_normalize(df.to_dict(orient='records'))
    df['departure.delay'] = df['departure.delay'].fillna(0)
    df['departure.scheduled_hour'] = pd.to_datetime(df['departure.scheduled']).dt.hour
    categorical_features = ['departure.airport', 'arrival.airport', 'airline.name']
    numerical_features = ['departure.scheduled_hour']
    features = df[categorical_features + numerical_features]
    features = column_transformer.transform(features)
    return features

def get_flight_data(access_key, flight_iata, filename='test_flight_data.jsonl', callback=None):
    base_url = 'http://api.aviationstack.com/v1/flights'
    params = {
        'access_key': access_key,
        'flight_iata': flight_iata,
        'arr_iata': 'SFO'
    }
    if callback:
        params['callback'] = callback
    response = requests.get(base_url, params=params)
    if callback:
        data = json.loads(response.text[len(callback) + 2 : -2])
    else:
        data = response.json()
    with open(filename, 'w') as outfile:
        for flight in data.get('data', []):
            json_record = json.dumps(flight)
            outfile.write(f"{json_record}\n")
    print(f"Flight data saved to {filename}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        flight_iata = request.form['flight_iata'].replace(' ', '')
        return predict(flight_iata)
    return render_template('index.html')

def predict(flight_iata):
    get_flight_data(api_key, flight_iata)
    test_df = pd.read_json('test_flight_data.jsonl', lines=True)
    X_test_new = transform_data(test_df, column_transformer)
    X_test_new_dense = X_test_new.toarray()
    X_test_new_tensor = torch.tensor(X_test_new_dense, dtype=torch.float32)
    predictions_new = model(X_test_new_tensor).detach().numpy()
    rounded_predictions = [f"{round(float(prediction))} minutes" for prediction in predictions_new.flatten()]

    return render_template('predictions.html', predictions=rounded_predictions, flight_iata=flight_iata)
    # return render_template('predictions.html', predictions=predictions_new.tolist(), flight_iata=flight_iata)

if __name__ == '__main__':
    app.run(debug=True)
