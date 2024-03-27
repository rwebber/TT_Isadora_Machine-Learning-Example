import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier

# iz_input 1 "training_data_json"
# iz_input 2 "train_trigger"
# iz_output 1 "training_status"

python_modules_path = os.path.join(os.path.dirname(__file__), '')
# MODEL_FILE = python_modules_path + 'trained_model.pkl'
# DATA_FILE = python_modules_path + 'training_data.json'

project_directory = os.path.abspath(os.path.join(python_modules_path, os.pardir))
MODEL_FILE = project_directory + "\\" + 'trained_model.pkl'
DATA_FILE = project_directory + "\\" + 'training_data.json'

TRAINING_STATUS = False

def save_data(data_json):
    """
    Saves the provided data in JSON format to a file.

    Parameters:
    data_json (str): The data to be saved, in JSON format.

    Returns:
    str: A message indicating the result of saving the data. Possible values are:
         - "Invalid data format. Expected JSON with 'features' and 'label'." if the data is not in the expected format.
         - "Data saved" if the data was successfully saved.
         - "Invalid JSON format" if the provided JSON data is not valid.
    """
    try:
        data = json.loads(data_json)
        if not isinstance(data, dict) or 'features' not in data or 'label' not in data:
            return "Invalid data format. Expected JSON with 'features' and 'label'."
        with open(DATA_FILE, 'a') as file:
            json.dump(data, file)  # creates new file if not already available
            file.write('\n')
        return "Data saved"
    except json.JSONDecodeError:
        return "Invalid JSON format"

def load_training_data():
    """
    Load training data from a file.

    Returns:
        A list[] of training data, where each element is a JSON object.
        Errors are returned as Strings.

    Raises:
        TypeError: If the file cannot be opened or read.
        json.JSONDecodeError: If the JSON format in the file is invalid.
    """
    try:
        with open(DATA_FILE, 'r') as file:
            return [json.loads(line) for line in file]
    except json.JSONDecodeError:
        return "Error loading training data: Invalid JSON format in file"
    except Exception as e:
        return f"Error loading training data: {e}"

def train_model():
    """
    Train the model using the loaded training data.

    Returns:
        str: A message indicating the status of the training process.
             - If the training is completed successfully, returns "Training completed".
             - If there is an error during training, returns the error message.
    """
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
    """
    Parameters:
    - training_data_json: a JSON object containing the training data
    - train_trigger: a boolean indicating whether to trigger the training process

    Return type:
    - If the train_trigger is True and the TRAINING_STATUS is False, the method returns:
        - If saving the training data is successful, it returns the string "Data saved"
        - If saving the training data fails, it returns a response indicating the failure
    - If the train_trigger is True and the TRAINING_STATUS is True, the method returns the string "Model already trained"
    - If the train_trigger is False, the method returns the result of saving the training data

    """
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
    print(python_main('{"features":[-3.964340812451968, -22.494763392023742], "label":"cat2"}', True))
