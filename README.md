💳 Credit Card Fraud Detection — MLOps Project

This project demonstrates the transformation of a traditional machine learning experiment into a production-oriented MLOps workflow.

The goal is to detect fraudulent credit card transactions and simulate the full lifecycle of a machine learning system, including experiment tracking, model selection, versioning, and deployment simulation.

🎯 Project Objective

The project evolves a previous academic ML project into a structured and reproducible engineering workflow.

Main goals:

Organize experiments into reusable scripts
Track experiments using MLflow
Compare multiple models and dimensionality reduction techniques
Select the best model based on technical and business metrics
Simulate deployment and inference of the final model
📊 Dataset

Dataset used: Credit Card Fraud Detection

The dataset contains anonymized transaction features (V1 – V28), plus:

Time
Amount
Class (0 = Legit, 1 = Fraud)

Highly imbalanced dataset:

Fraud transactions ≈ 0.17%

This makes recall and precision critical metrics.

⚙️ Project Structure
project/
│
├── script.py          # Training + MLflow experiment tracking
├── predict.py         # Model loading and inference simulation
├── requirements.txt
└── README.md
🧠 Models Evaluated

Baseline models:

Decision Tree
Random Forest

Dimensionality reduction:

PCA (Principal Component Analysis)
LDA (Linear Discriminant Analysis)

Experiments tracked using MLflow.

🏆 Final Model Selection

Best performing model:

👉 Random Forest + PCA

Why?

Best balance between precision and recall
Good generalization
Reduced dimensionality → lower computational cost
Suitable for production deployment
📈 Experiment Tracking with MLflow

MLflow was used to track:

Model parameters
Evaluation metrics (Accuracy, Precision, Recall, F1)
Model artifacts (trained models)
Experiment comparison

To start MLflow UI:

mlflow ui

Open in browser:

http://127.0.0.1:5000
▶️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Start MLflow server
mlflow ui
3️⃣ Train models
python script.py

This will:

Train multiple models
Track experiments
Save trained models in MLflow
4️⃣ Run inference simulation
python predict.py

This simulates the model being used in production.

🚀 Production Simulation

The project includes:

Model versioning
Artifact storage
Inference pipeline
Monitoring-ready workflow

This simulates a real MLOps deployment lifecycle.

👩‍💻 Author

Fernanda — Machine Learning Engineering Student
