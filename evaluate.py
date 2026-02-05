import json
import os
import tarfile
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    print("Starting evaluation script...")

    # 1. Define Paths (Standard SageMaker Processing Paths)
    # These are where SageMaker mounts your data and model inside the container
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/test.csv"
    output_path = "/opt/ml/processing/evaluation/evaluation.json"

    # 2. Load Test Data
    print(f"Reading test data from {test_path}")
    df_test = pd.read_csv(test_path)
    
    # Separate Features and Target (Assuming 'Class' is the target column)
    y_test = df_test['Class']
    X_test = df_test.drop('Class', axis=1)

    # 3. Load Model
    # The training job outputs a 'model.tar.gz'. We need to extract it first.
    print(f"Extracting model from {model_path}")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    print("Loading model.joblib...")
    model = joblib.load("model.joblib")

    # 4. Make Predictions
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # 5. Calculate Metrics
    # We calculate the standard metrics for fraud detection
    print("Calculating metrics...")
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # 6. Save Evaluation Report (Critical Step)
    # This specific JSON format is required so SageMaker can parse it later
    report_dict = {
        "binary_classification_metrics": {
            "recall": {
                "value": recall,
                "standard_deviation": "NaN"
            },
            "precision": {
                "value": precision,
                "standard_deviation": "NaN"
            },
            "f1": {
                "value": f1,
                "standard_deviation": "NaN"
            },
            "accuracy": {
                "value": accuracy,
                "standard_deviation": "NaN"
            },
            "auc": {
                "value": auc,
                "standard_deviation": "NaN"
            }
        }
    }

    print(f"Saving evaluation report to {output_path}")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    print("Evaluation complete.")
