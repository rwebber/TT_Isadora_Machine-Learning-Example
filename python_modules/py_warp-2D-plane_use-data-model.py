import json
import joblib

# iz_input 1 "input_data_json"
# iz_output 1 "predicted_position"

MODEL_FILE = 'trained_model.pkl'

def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        return None

def python_main(input_data_json):
    model = load_model()
    if model is None:
        return "Model not found."
    try:
        data = json.loads(input_data_json)
        if 'input' not in data:
            return "Invalid input format."
        prediction = model.predict([data['input']])
        return json.dumps({'predicted_position': prediction[0].tolist()})
    except json.JSONDecodeError:
        return "Invalid JSON format"

if __name__ == '__main__':
    # This section is for IDE development and won't run inside Pythoner
    test_input = '{"input": [-1, 5]}'
    print(python_main(test_input))
