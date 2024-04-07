import requests
import json
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access your API key
api_key = os.getenv('API_KEY')

def get_flight_data(access_key, filename='flight_data.jsonl', callback=None):
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

# Usage example (replace 'YOUR_ACCESS_KEY' with your actual API key)
access_key = api_key
get_flight_data(access_key)

