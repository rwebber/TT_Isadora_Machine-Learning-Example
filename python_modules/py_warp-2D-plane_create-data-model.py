import os
import json
import joblib
from sklearn.ensemble import RandomForestRegressor

# iz_input 1 "training_data_json"
# iz_input 2 "train_trigger"
# iz_output 1 "training_status"

# Determine paths relative to the script location
python_modules_path = os.path.join(os.path.dirname(__file__), '')
project_directory = os.path.abspath(os.path.join(python_modules_path, os.pardir))
MODEL_FILE = os.path.join(project_directory, 'trained_model.pkl')
DATA_FILE = os.path.join(project_directory, 'training_data.json')

def save_data(data_json):
    """
    Saves the provided data in JSON format to a file.
    """
    try:
        data = json.loads(data_json)
        if not isinstance(data, list) or any('input' not in d or 'label' not in d for d in data):
            return "Invalid data format."
        with open(DATA_FILE, 'w') as file:
            json.dump(data, file)
        return "Data saved"
    except json.JSONDecodeError:
        return "Invalid JSON format"

def train_model():
    """
    Train the model using the saved training data.
    """
    try:
        with open(DATA_FILE, 'r') as file:
            data = json.load(file)
        inputs = [item['input'] for item in data]
        labels = [item['label'] for item in data]
        model = RandomForestRegressor()
        model.fit(inputs, labels)
        joblib.dump(model, MODEL_FILE)
        return "Training completed"
    except Exception as e:
        return f"Error during training: {e}"

def python_main(training_data_json, train_trigger):
    if train_trigger:
        response = save_data(training_data_json)
        if response != "Data saved":
            return response
        return train_model()
    return "No action taken"

if __name__ == '__main__':
    # This section is for IDE development and won't run inside Pythoner
    test_data = '[{"input": [-30, -33], "label": [-50, -50]}, {"input": [-2, 5], "label": [0, 0]}]'
    print(python_main(test_data, True))
