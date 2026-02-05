# AWS MLOps: Automated Fraud Detection Pipeline

**End-to-end MLOps pipeline for Credit Card Fraud Detection, featuring automated retraining, drift detection, and CI/CD deployment using Amazon SageMaker.**

## 🚀 Project Overview
This project solves a critical problem for government agencies: manual fraud detection processes cannot scale to handle high transaction volumes or adapt to changing fraud patterns (Data & Concept Drift).

I built a fully automated **SageMaker Pipeline** that ingests anonymized transaction data, trains a Logistic Regression model (optimized for high recall), and deploys it to a production REST endpoint. The system includes a "Drift Trigger" that automatically retrains the model when statistical properties of the data change.

## 🏗️ Architecture
The system follows a mature MLOps pattern using AWS serverless components:

1.  **Data Lake (S3):** Acts as the single source of truth for raw transaction logs and model artifacts.
2.  **SageMaker Pipeline:** Orchestrates the workflow:
    * *Preprocessing:* Binning, SMOTE upsampling (to fix 0.17% class imbalance), and standard scaling.
    * *Training:* Logistic Regression (chosen for low Type I error cost).
    * *Evaluation:* Quality Gate that only registers models if Recall > 0.60.
3.  **Model Registry:** Versions approved models and triggers deployment.
4.  **Deployment (CI/CD):** An **EventBridge** rule triggers an AWS Lambda function to deploy approved models to a real-time SageMaker Endpoint.
5.  **Monitoring:** A Model Monitor captures live traffic to detect drift and trigger retraining loops.

![Pipeline Diagram](assets/pipeline_diagram.jpg)

## 📊 Key Results & "Drift" Simulation
To test the pipeline's robustness, I simulated a 12-month production scenario involving both **Data Drift** (shifting feature means) and **Concept Drift** (flipping labels for high-value transactions).

* **Outcome:** The system successfully detected the performance degradation (Recall drop) and triggered an automated retraining job, restoring model performance without manual intervention.
* **Performance:** The final model achieved a Recall of **0.90**, ensuring maximum detection of fraudulent cases.

## 🔧 Engineering Challenges
**1. The "Dependency Hell" S3 Error**
* *Issue:* The training script failed with a `NoneType can't be used in 'await'` error.
* *Root Cause:* Installing `awscli` auto-upgraded `botocore`, breaking the `aiobotocore` library used by Pandas for S3 connections.
* *Solution:* Refactored the environment configuration to exclude the conflicting `awscli` installation.

**2. Root vs. IAM Permissions**
* *Issue:* `Unknown IAM PrincipalArn: root` error during project creation.
* *Solution:* Enforced strict IAM user separation via browser profiles to prevent accidental Root user execution.

## 🛠️ Tech Stack
* **Cloud Platform:** AWS SageMaker (Pipelines, Model Registry, Endpoints).
* **Orchestration:** Amazon EventBridge (Trigger), AWS Lambda (Deployment).
* **Storage & Monitoring:** Amazon S3, CloudWatch, SageMaker Model Monitor.
* **Machine Learning:** Scikit-Learn (Logistic Regression, SMOTE), Pandas.
* **Environment:** Python 3.10 (SageMaker Managed Runtimes).

## 📂 Repository Structure
* `pipelines/`: Contains the `pipeline.py` definition for SageMaker.
* `scripts/`: Python scripts for `preprocessing.py`, `train.py`, and `evaluate.py`.
* `lambda_deploy/`: The Lambda function code for endpoint deployment.
* `notebooks/`: Exploratory Data Analysis (EDA) and drift simulation experiments.

## 👤 Author
**Brandyn Ewanek**
* [LinkedIn](https://www.linkedin.com/in/brandyn-ewanek-9733873b/)
* [Portfolio](https://github.com/Brandyn-Ewanek/)
