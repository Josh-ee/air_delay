import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ast
import json

from joblib import dump

def load_and_preprocess_data(filepath, column_transformer=None, fit_transformer=True):
    df = pd.read_json(filepath, lines=True)
    df = pd.json_normalize(df.to_dict(orient='records'))
    df['departure.delay'] = df['departure.delay'].fillna(0)
    df['departure.scheduled_hour'] = pd.to_datetime(df['departure.scheduled']).dt.hour

    categorical_features = ['departure.airport', 'arrival.airport', 'airline.name']
    numerical_features = ['departure.scheduled_hour']
    features = df[categorical_features + numerical_features]
    target = df['departure.delay']

    if column_transformer is None:
        column_transformer = ColumnTransformer([
            ('ohe', OneHotEncoder(), categorical_features),
            ('scaler', StandardScaler(), numerical_features)
        ])
        if fit_transformer:
            features = column_transformer.fit_transform(features)
        else:
            features = column_transformer.transform(features)
    else:
        assert not fit_transformer, "Column transformer should not be fitted again."
        features = column_transformer.transform(features)

    return features, target, column_transformer


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Train the model
def train_model(X_train, y_train, input_size, num_epochs=1000, batch_size=10, learning_rate=0.01):
    model = LinearRegressionModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Convert X_train to a dense array if it's a sparse matrix
    if isinstance(X_train, np.ndarray):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    else:  # Assuming X_train is a sparse matrix
        X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    return model

features, target, column_transformer = load_and_preprocess_data('flight_data.jsonl')
# print(features)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# print(X_test)
model = train_model(X_train, y_train, input_size=X_train.shape[1])

# Predict and evaluate
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32) if not isinstance(X_test, np.ndarray) else torch.tensor(X_test, dtype=torch.float32)
predictions = model(X_test_tensor).detach().numpy()

# Evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')



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

# Assume we have loaded and transformed the training data earlier
# Train model etc.

# print(column_transformer)

# Save the model
dump(model, 'linear_regression_model.joblib')

# Save the column transformer
dump(column_transformer, 'column_transformer.joblib')

torch.save(model.state_dict(), 'linear_regression_model.pth')


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
