import argparse
import os
import pandas as pd
import joblib
import sys
import subprocess

# --- INSTALL DEPENDENCY ---
subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    print("Starting training script...")

    # 1. Parse SageMaker Arguments
    parser = argparse.ArgumentParser()
    
    # Standard SageMaker args
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    
    # Pipeline args
    parser.add_argument("--max_iter", type=int, default=1000)
    
    # SAFETY FIX: Use parse_known_args() instead of parse_args()
    # This prevents the "ExitCode 2" crash if SageMaker passes extra hidden arguments
    args, _ = parser.parse_known_args()

    # 2. Load Training Data
    train_file = os.path.join(args.train, "train.csv")
    print(f"Reading training data from {train_file}")
    train_df = pd.read_csv(train_file)

    # Separate Features and Target
    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]

    # 3. Apply SMOTE
    print("Applying SMOTE resampling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original shape: {X_train.shape}, Resampled shape: {X_resampled.shape}")

    # 4. Create Pipeline
    print(f"Training Pipeline with max_iter={args.max_iter}...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=args.max_iter))
    ])

    # 5. Train
    pipeline.fit(X_resampled, y_resampled)

    # 6. Save Model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
