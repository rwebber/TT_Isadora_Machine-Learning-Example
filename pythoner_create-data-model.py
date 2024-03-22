import json
import joblib
from sklearn.ensemble import RandomForestClassifier

# iz_input 1 "training_data_json"
# iz_input 2 "train_trigger"
# iz_output 1 "training_status"

MODEL_FILE = 'trained_model.pkl'
DATA_FILE = 'training_data.json'
TRAINING_STATUS = False

def save_data(data_json):
    try:
        data = json.loads(data_json)
        if not isinstance(data, dict) or 'features' not in data or 'label' not in data:
            return "Invalid data format. Expected JSON with 'features' and 'label'."
        with open(DATA_FILE, 'a') as file:
            json.dump(data, file)
            file.write('\n')
        return "Data saved"
    except json.JSONDecodeError:
        return "Invalid JSON format"

def load_training_data():
    try:
        with open(DATA_FILE, 'r') as file:
            return [json.loads(line) for line in file]
    except json.JSONDecodeError:
        return "Error loading training data: Invalid JSON format in file"
    except Exception as e:
        return f"Error loading training data: {e}"

def train_model():
    data_load_response = load_training_data()
    if isinstance(data_load_response, str):
        return data_load_response  # Return error message if any

    try:
        features = [item['features'] for item in data_load_response]
        labels = [item['label'] for item in data_load_response]
        model = RandomForestClassifier()
        model.fit(features, labels)
        joblib.dump(model, MODEL_FILE)
        global TRAINING_STATUS
        TRAINING_STATUS = True
        return "Training completed"
    except Exception as e:
        return f"Error during training: {e}"

# python_main is called whenever an input value changes
def python_main(training_data_json, train_trigger):
    if train_trigger and not TRAINING_STATUS:
        response = save_data(training_data_json)
        if response != "Data saved":
            return response
        return train_model()
    elif train_trigger:
        return "Model already trained"
    else:
        return save_data(training_data_json)

if __name__ == '__main__':
    # For testing outside Isadora
    print(python_main('{"features":[-3.964240802451968, -22.494763392023742], "label":"cat2"}', True))
