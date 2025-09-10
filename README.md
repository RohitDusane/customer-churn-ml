
# 🔍 Customer Churn Prediction Pipeline (End-to-End ML Project)

This project demonstrates a full-scale machine learning pipeline to predict customer churn using Python, scikit-learn, XGBoost, SHAP, and MLflow. It includes data ingestion, validation, transformation, model training, evaluation, and deployment via **FastAPI**.

---

## 🚀 Key Features

✅ Modularized ML Pipeline using OOP & Configurations  
✅ Automated Logging via Python's `logging` and `MLflow`  
✅ Baseline model comparison (RandomForest, XGBoost, LogisticRegression, etc.)  
✅ Advanced metrics: Top-Decile Lift, Overfitting Warning  
✅ SHAP-based model explainability  
✅ FastAPI for real-time inference API  

---

## 🛠️ Project Structure

```
Churn_Pred/
├── components/              # Data ingestion, validation, transformation modules
├── entity/                  # Artifact data classes
├── logger/                  # Logging module
├── exception/               # Custom exception handling
├── utils/                   # Utility functions
├── artifacts/               # Outputs: models, reports, metrics
├── app/                     # FastAPI app for deployment
├── main.py                  # Training pipeline entry point
├── mlruns/                  # MLflow tracking
└── README.md
```
---


## 🔁 ML Pipeline Flow

```bash
📦 Data Ingestion → ✅ Validation → 🔄 Transformation → 🔧 Model Training → 📈 Evaluation → 🌐 FastAPI
```

---

## 🔍 Pipeline Stages

### 📦 Data Ingestion
- Reads raw data and drops irrelevant columns (e.g., IDs, names)
- Splits into training and test sets

### ✅ Data Validation
- Checks schema consistency, missing values, and data drift
- Ensures all required columns are present

### 🔄 Data Transformation
- Applies `ColumnTransformer` pipelines for numeric and categorical features
- Saves:
  - `.npy` arrays (for model training)
  - `.csv` DataFrames (for debugging)
  - `preprocessor.pkl` (for deployment reuse)

### 🔧 Model Training
- Trains baseline models (RandomForest, XGBoost, LogisticRegression, etc.)
- Evaluates using:
  - **F1 Score**
  - **ROC AUC**![ROC](https://github.com/RohitDusane/customer-churn-ml/blob/main/Churn_Pred/screenshot/XGBoost_roc_curve.png)
  - **Top-Decile Lift**
  - **Feature Importance** ![Feature Importance](https://github.com/RohitDusane/customer-churn-ml/blob/main/Churn_Pred/screenshot/XGBoost_feature_importance.png)
- Tracks all experiments via **MLflow**
- Selects and saves the best-performing model

### 🌐 API Deployment
- Exposes real-time inference endpoint using **FastAPI**
- API UI available at: `http://localhost:8000/`
- Can be containerized via **Docker** for scalable deployment

---

## 📊 Metrics Tracked

- Accuracy, F1 Score, Precision, Recall
- ROC AUC
- Cross-validation F1 score (mean & std)
- Top-Decile Lift
- Overfitting/Underfitting Warnings

---

## 📦 Installation

```bash
pip install -r requirements.txt
```
---

## 🧪 Train the Pipeline

```bash
python main.py
```
---


## 🌐 Running API Server

```bash
uvicorn app:app --reload
```

Access App at: `http://localhost:8000`


---

---
## 📁 Artifacts & Logs

* Metrics, plots, confusion matrices
* Trained models stored in: `artifacts/model_trained`
* MLflow dashboard: `http://localhost:5000` (in terminal `mlflow ui`) 

## Steps to View the MLflow Dashboard Locally:

### 1. Start MLflow UI:

- Open a terminal (or command prompt).
- Navigate to your project folder (e.g., `Churn_Pred`).
- Run the command:

  ```bash
  mlflow ui
  ```

### 2. Access the Dashboard:

- Open a browser and go to http://localhost:5000 ![MLflow Dashboard](https://github.com/RohitDusane/customer-churn-ml/blob/main/Churn_Pred/screenshot/mlflow dashbaord.png)
- This will open the MLflow UI where you can see all of your logged experiments, parameters, metrics, and models.

---

## 📸 Example Output

![conf_matrix](https://github.com/RohitDusane/customer-churn-ml/blob/main/Churn_Pred/screenshot/XGBoost_conf_matrix.png)


---


## 💡 SMOTETomek Explainability

SMOTETomek is a powerful tool to address class imbalance by both increasing the minority class size and cleaning up the dataset by removing noisy or borderline examples. By doing so, it enables models to perform better in imbalanced settings, leading to improved generalization and more accurate predictions. It is especially useful in tasks like fraud detection, medical diagnosis, and anomaly detection where minority class instances are critical for model performance but are often underrepresented.

![Before SMOTETomek](https://github.com/RohitDusane/customer-churn-ml/blob/main/Churn_Pred/screenshot/Before SMOTETomek.png)
![After SMOTETomek](https://github.com/RohitDusane/customer-churn-ml/blob/main/Churn_Pred/screenshot/After Smotetomek.png)


---


## 🧠 Best Model

By default, the best model is selected based on **F1 Score** and saved for deployment.


---


## 📍 Future Improvements

* Model versioning via DVC
* CI/CD for automated testing and deployment
* Advanced hyperparameter tuning with Optuna


---


## 👨‍💻 Author

* **Rohit Dusane**
* [LinkedIn](https://www.linkedin.com/in/rohit-dusane/) | [GitHub](https://github.com/RohitDusane/)


---

## 🏁 License

MIT License

---
