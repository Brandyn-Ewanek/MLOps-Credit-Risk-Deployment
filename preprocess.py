import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


    
# 1. Define Standard SageMaker Container Paths
# SageMaker mounts your S3 bucket to these local folders automatically
base_dir = "/opt/ml/processing"
input_path = os.path.join(base_dir, "input", "creditcard.csv")

# 2. Load Data
print(f"Reading data from {input_path}")
df = pd.read_csv(input_path)

# 3. Feature Selection
# Drop 'Time' and 'Class' to separate Features (X) and Target (y)
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

# 4. Split Data
# We use the same random_state=42 to ensure consistency
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Recombine for Saving
# The training script expects a CSV with the target 'Class' included
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# 6. Save Processed Files
# SageMaker will automatically upload everything in these folders to S3
train_output_path = os.path.join(base_dir, "train", "train.csv")
test_output_path = os.path.join(base_dir, "test", "test.csv")

print(f"Saving train data to {train_output_path}")
train_df.to_csv(train_output_path, index=False)

print(f"Saving test data to {test_output_path}")
test_df.to_csv(test_output_path, index=False)

print("Preprocessing complete!")
