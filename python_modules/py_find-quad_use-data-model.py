import os
import json
import joblib

# iz_input 1 "input_data_json"
# iz_output 1 "prediction/status"

# Define the path to the machine learning model file
MODEL_FILE = 'trained_model.pkl'
# Initialize a global variable for the model to manage its lifecycle
model = None

def get_model_path():
    """
    Determine the correct path to the model file.
    The path depends on whether the script is running within Pythoner in Isadora
    or in a standard Python environment.
    """
    # Check if the script is running in Pythoner (Isadora)
    if __name__ != '__main__':
        # Construct the path assuming the model file is in the same directory as the Isadora project
        python_modules_path = os.path.join(os.path.dirname(__file__), '')
        project_directory = os.path.abspath(os.path.join(python_modules_path, os.pardir))
        return os.path.join(project_directory, MODEL_FILE)
    else:
        # If running in a standard Python environment, assume the model is in the current directory
        return MODEL_FILE

def load_model():
    """
    Load the machine learning model from the file.
    This function sets the global model variable.
    """
    global model
    model_path = get_model_path()

    if not os.path.exists(model_path):
        return "Model file not found at: " + model_path

    try:
        model = joblib.load(model_path)
        return "Model loaded successfully"
    except Exception as e:
        return f"Error loading model: {e}"

# python_main is called whenever an input value changes
def python_main(classification_data_json):
    """
    Main function called by Pythoner for classification.
    It loads the model if not already loaded, processes the input data,
    and performs classification using the model.
    """
    global model
    if model is None:
        message = load_model()
        if model is None:
            return message

    try:
        # Parse the input JSON data
        data = json.loads(classification_data_json)
        if not isinstance(data, dict) or 'features' not in data:
            return "Invalid data format. Expected JSON with 'features'."
        
        # Perform prediction using the model
        prediction = model.predict([data['features']])
        return json.dumps({'prediction': prediction[0]})
    except json.JSONDecodeError:
        return "Invalid JSON format"

# python_finalize is called just before the actor is deactivated
def python_finalize():
    """
    Finalize function called by Pythoner before deactivation.
    It releases the model to ensure clean resource management.
    """
    global model
    model = None  # Release the model
    print("Model and resources released")

if __name__ == '__main__':
    # Testing the script outside Isadora
    test_data = '{"features":[-3.964340812451968, -22.494763392023742]}'
    print(python_main(test_data))  # Perform a test prediction
    python_finalize()  # Clean up resources
