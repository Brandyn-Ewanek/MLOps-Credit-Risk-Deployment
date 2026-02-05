import joblib
import os
import json
import numpy as np

# 1. model_fn: Loads the model from disk
def model_fn(model_dir):
    """
    Load the model from the directory where SageMaker saved it.
    """
    print("Loading model from: {}".format(model_dir))
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

# 2. input_fn: Deserializes the input data
def input_fn(request_body, request_content_type):
    """
    Parse the incoming request. We support text/csv.
    """
    if request_content_type == 'text/csv':
        # Read the raw CSV string and convert to a list of floats
        # This handles a single line of CSV like "0.1, 1.2, -0.5, ..."
        data = [float(x) for x in request_body.strip().split(',')]
        # Scikit-learn expects a 2D array: [[f1, f2, f3...]]
        return np.array([data])
    else:
        # You can add logic for 'application/json' if needed
        raise ValueError(f"Unsupported content type: {request_content_type}")

# 3. predict_fn: Makes the prediction
def predict_fn(input_data, model):
    """
    Apply the model to the input data.
    """
    # We use predict_proba to get the probability of Fraud (class 1)
    # This gives us a score (e.g., 0.85) rather than just a 0/1 label
    prediction = model.predict_proba(input_data)
    
    # prediction is usually [[prob_0, prob_1]]. We want prob_1.
    return prediction[0][1]

# 4. output_fn: Serializes the prediction result
def output_fn(prediction, content_type):
    """
    Format the output for the client.
    """
    # Return the probability as a simple JSON object
    response = {'fraud_probability': prediction}
    return json.dumps(response)
