from joblib import load
import pandas as pd
import torch.nn as nn
import torch
import json
import requests
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access your API key
api_key = os.getenv('API_KEY')


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Load the model
model = LinearRegressionModel(81)

# Load the model parameters
model.load_state_dict(torch.load('linear_regression_model.pth'))

# Load the column transformer
column_transformer = load('column_transformer.joblib')


# Load the column transformer
column_transformer = load('column_transformer.joblib')

def transform_data(df, column_transformer):
    # Flatten nested dictionaries
    df = pd.json_normalize(df.to_dict(orient='records'))

    # Fill NaN delay values with zero
    df['departure.delay'] = df['departure.delay'].fillna(0)

    # Convert 'scheduled' departure time to datetime and extract the hour
    df['departure.scheduled_hour'] = pd.to_datetime(df['departure.scheduled']).dt.hour

    # Select features
    categorical_features = ['departure.airport', 'arrival.airport', 'airline.name']
    numerical_features = ['departure.scheduled_hour']
    features = df[categorical_features + numerical_features]

    # Transform features using the existing transformer
    features = column_transformer.transform(features)
    return features



def get_flight_data(access_key, filename='test_flight_data.jsonl', callback=None):
    """
    Fetch flight data from the Aviationstack API and save it as a JSONL file.

    :param access_key: API access key as a string.
    :param filename: The filename to which the data will be saved.
    :param callback: Optional callback function name for JSONP response.
    """
    base_url = 'http://api.aviationstack.com/v1/flights'
    
    params = {
        'access_key': access_key,
        # 'dep_iata': 'SFO',  # Filtering for flights departing from SFO
        'flight_iata': 'UA5583', # UA5583
        'arr_iata': 'SFO'
    }
    
    if callback:
        params['callback'] = callback
    
    response = requests.get(base_url, params=params)
    
    # Process and save the response in JSONL format
    if callback:
        # Remove the JSONP callback syntax to extract the JSON part
        data = json.loads(response.text[len(callback) + 2 : -2])
    else:
        data = response.json()

    # Write each flight record on a new line in the file
    with open(filename, 'w') as outfile:
        for flight in data.get('data', []):  # Ensure the 'data' key exists
            json_record = json.dumps(flight)
            outfile.write(f"{json_record}\n")

    print(f"Flight data saved to {filename}")

access_key = api_key
get_flight_data(access_key)


# When it's time to predict, we do the following:
test_df = pd.read_json('test_flight_data.jsonl', lines=True)


X_test_new = transform_data(test_df, column_transformer)  # Use the transformer fitted on the training data

# Convert to tensor and predict
# X_test_new_tensor = torch.tensor(X_test_new, dtype=torch.float32)

X_test_new_dense = X_test_new.toarray()  # Convert sparse matrix to dense
X_test_new_tensor = torch.tensor(X_test_new_dense, dtype=torch.float32)  # Now convert to tensor

# Now you can proceed with using your model to make predictions
predictions_new = model(X_test_new_tensor).detach().numpy()

# Output predictions
print("Predicted Delays:")
print(predictions_new)
