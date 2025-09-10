import os, sys
import joblib
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score, classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Churn_Pred.exception.exception import CustomException
from Churn_Pred.logger.log import logging
from Churn_Pred.components.data_ingestion import DataIngestionConfig, DataIngestion
from Churn_Pred.components.data_validation import DataValidationConfig, DataValidation
from Churn_Pred.components.data_transformation import DataTransformationConfig, DataTransformation
from Churn_Pred.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact
from Churn_Pred.utils import get_lift_status, get_overfit_warning


import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import dagshub  # tracking os million oof experiment virtually via onnecting the github repository

dagshub.init(repo_owner='stat.data247', repo_name='customer-churn-ml', mlflow=True)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from collections import Counter

import mlflow

experiment_name = "ChurnPredictionModels"
# Check if the experiment exists, if not, create it
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)


# ====== CONFIG CLASS ======
@dataclass
class ModelTrainerConfig:
    trained_model_dir: Path = Path('artifacts/model_trained/best_models/')
    trained_model_filepath: Path = Path('artifacts/model_trained/models/model.pkl')
    train_metric_filepath: Path = Path('artifacts/model_trained/metrics/train.npy')
    test_metric_filepath: Path = Path('artifacts/model_trained/metrics/test.npy')
    trained_conf_matrix: Path = Path('artifacts/model_trained/images')
    trained_roc_plot: Path = Path('artifacts/model_trained/images')
    trained_shap_dir: Path = Path("artifacts/model_trained/shap")
    important_feat_csv_path = Path("artifacts/model_trained/metrics/top_features.csv")
    important_feat_plot: Path = Path('artifacts/model_trained/images')
    lift_gain_csv_path: Path = Path('artifacts/model_trained/metrics/lift')
    expected_accuracy: float = 0.6
    overfit_underfit_threshold: float = 0.05


# ====== MODEL TRAINER CLASS ======
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact

    # def save_model_trainer_artifact(self, artifact: ModelTrainerArtifact, filepath: str):
    #     with open(filepath, 'w') as f:
    #         json.dump(artifact.__dict__, f, indent=4)
    #     print(f"ModelTrainerArtifact saved at {filepath}")



    def train_baseline_models(self):
        logging.info('🚀 Starting Baseline Model Training \n')
        try:
            # Load Data
            preprocessor = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)
            train_df = pd.read_csv(self.data_transformation_artifact.transformed_train_df_filepath)
            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_df_filepath)

            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            models = {
                'RandomForest': RandomForestClassifier(),
                'DecisionTree': DecisionTreeClassifier(),
                'LogisticRegression': LogisticRegression(),
                'XGBoost': XGBClassifier(eval_metric='logloss')
            }

            report = []
            trained_pipelines = {}
            feature_dfs = []

            # Create base dataframe to collect probabilities
            prob_df = X_test.copy()
            prob_df['actual'] = y_test.values

            for model_name, model in models.items():
                with mlflow.start_run(run_name=f"Baseline-{model_name}"):
                    logging.info(f"\t🔧 Training {model_name} (Baseline)")
                    pipeline = ImbPipeline([
                        ('sampling', SMOTETomek(random_state=42)),
                        ('classifier', XGBClassifier(eval_metric='logloss'))
                    ])
                    pipeline.fit(X_train, y_train)

                    

                    y_train_pred = pipeline.predict(X_train)
                    y_test_pred = pipeline.predict(X_test)

                    # Add predicted probabilities for the current model
                    prob_df[f"{model_name}_base_prob"] = pipeline.predict_proba(X_test)[:, 1]

                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)

                    class_report = classification_report(y_test, y_test_pred, output_dict=True)
                    f1 = class_report['1'].get('f1-score', 0)
                    precision = class_report['1'].get('precision', 0)
                    recall = class_report['1'].get('recall', 0)
                    roc_auc = self.calculate_roc_auc(pipeline, X_test, y_test) 

                    output_dir = Path('artifacts/model_trained/metrics/lift')
                    output_dir.mkdir(parents=True, exist_ok=True)

                    lift_score = self.plot_lift_gain(model_name, pipeline, X_test, y_test)
                    overfit_warning = get_overfit_warning(model_name, train_acc, test_acc, self.config.overfit_underfit_threshold)
                    lift_status = get_lift_status(model_name, lift_score)

                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')

                    cm = confusion_matrix(y_test, y_test_pred)
                    self.save_all_model_outputs(model_name, pipeline, cm, roc_auc, X_test, y_test)
        

                    mlflow.log_metric("Train Accuracy", train_acc)
                    mlflow.log_metric("Test Accuracy", test_acc)
                    mlflow.log_metric("F1 Score", f1)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("ROC AUC", roc_auc)
                    mlflow.log_metric("CV F1 Mean", cv_scores.mean())
                    mlflow.log_metric("CV F1 Std", cv_scores.std())
                    mlflow.log_metric("Lift Score", lift_score)
                    mlflow.log_param("Lift Status", lift_status)  
                    mlflow.log_param("Overfit/Underfit Warning",overfit_warning)

                    # mlflow.sklearn.log_model(pipeline, name="model")
                    
                    signature = infer_signature(X_test, y_test_pred)
                    # mlflow.sklearn.log_model(model, name="model", input_example=X_test.iloc[:5], signature=signature)
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        name="model",
                        input_example=X_test.iloc[:5],
                        signature=signature
                    )

                    report.append({
                        "Model": model_name,
                        "Train Accuracy": train_acc,
                        "Test Accuracy": test_acc,
                        "F1": f1,
                        "Precision": precision,
                        "Recall": recall,
                        "ROC AUC": roc_auc,
                        "CV Mean": cv_scores.mean(),
                        "CV Std": cv_scores.std(),
                        "Lift Score": lift_score,
                        "Overfit Warning": overfit_warning,
                        "Lift Status": lift_status
                    })

                    trained_pipelines[model_name] = pipeline
            
                    # Inside loop per model:
                    feature_df = self.plot_feature_importance(model=pipeline.named_steps['classifier'], X=X_test, model_name=model_name)
                    if feature_df is not None:
                        feature_dfs.append(feature_df)

            # Feature importance list
            if feature_dfs:
                all_feature_df = pd.concat(feature_dfs)
                important_feat_csv_path = Path("artifacts/model_trained/metrics/top_features.csv")
                important_feat_csv_path.parent.mkdir(parents=True, exist_ok=True)
                all_feature_df.to_csv(important_feat_csv_path, index=False)
                logging.info(f"📌 Top N feature importance saved a t {important_feat_csv_path}")

            # Save final probability DataFrame
            final_prob_csv = Path("artifacts/model_trained/metrics/test_predicted_probabilities.csv")
            os.makedirs(final_prob_csv.parent, exist_ok=True)
            prob_df.to_csv(final_prob_csv, index=False)
            logging.info(f"✅ All model probabilities saved at  {final_prob_csv}")

            # Save report
            report_df = pd.DataFrame(report)
            report_path = Path("artifacts/model_trained/metrics/baseline_model_comparison.csv")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_df.to_csv(report_path, index=False)

            logging.info(f"📊 Baseline comparison report saved : {report_path}")
            self.save_combined_excel(report_df)

            best_model_row = report_df.sort_values(by="F1", ascending=False).iloc[0]
            best_model_name = best_model_row["Model"]
            best_pipeline = trained_pipelines[best_model_name]
            

            # Define model save path
            trained_model_filepath = os.path.join(self.config.trained_model_dir, f"{best_model_name}_model_pusher.pkl")

            # Ensure the directory exists
            model_dir = os.path.dirname(trained_model_filepath)
            os.makedirs(model_dir, exist_ok=True)

            # Save the best model pipeline
            joblib.dump(best_pipeline, trained_model_filepath)
            logging.info(f"✅ Best model is '{best_model_name}' saved at {trained_model_filepath}")

            # Create ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_filepath=trained_model_filepath,
                train_metric_filepath=self.config.train_metric_filepath,
                test_metric_filepath=self.config.test_metric_filepath,
                trained_conf_matrix=self.config.trained_conf_matrix,
                trained_roc_plot=self.config.trained_roc_plot,
                trained_shap_dir=self.config.trained_shap_dir,
                expected_accuracy=self.config.expected_accuracy,
                overfit_underfit_threshold=self.config.overfit_underfit_threshold,
                model_name=best_model_name,
                train_accuracy=best_model_row['Train Accuracy'],
                test_accuracy=best_model_row['Test Accuracy'],
                f1_score=best_model_row['F1'],
                precision=best_model_row['Precision'],
                recall=best_model_row['Recall'],
                roc_auc=best_model_row['ROC AUC']
            )

            return trained_pipelines, report_df, model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)



    def train_tuned_models(self):
        logging.info('🔍 Starting Hyperparameter Tuning...')
        try:
            # Load data
            preprocessor = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)
            train_df = pd.read_csv(self.data_transformation_artifact.transformed_train_df_filepath)
            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_df_filepath)

            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            tuned_models = {
                "RandomForest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "n_estimators": [50, 100],
                        "max_depth": [5, 10]
                    }
                },
                "DecisionTree": {
                    "model": DecisionTreeClassifier(),
                    "params": {
                        "max_depth": [3, 5, 10]
                    }
                },
                "LogisticRegression": {
                    "model": LogisticRegression(max_iter=1000),
                    "params": {
                        "C": [0.01, 0.1, 1, 10]
                    }
                },
                "XGBoost": {
                    "model": XGBClassifier(eval_metric='logloss'),
                    "params": {
                        "n_estimators": [50, 100],
                        "max_depth": [3, 6]
                    }
                }
            }

            report = []
            trained_pipelines = {}
            feature_dfs = []

            # Create probability dataframe
            prob_df = X_test.copy()
            prob_df['actual'] = y_test.values

            for model_name, config in tuned_models.items():
                with mlflow.start_run(run_name=f"Tuned-{model_name}"):
                    logging.info(f"\t🔧 Tuning and Training {model_name}")
                    # grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='f1', n_jobs=-1)
                    pipeline = ImbPipeline([
                        ('sampling', SMOTETomek(random_state=42)),
                        ('classifier', config["model"])
                    ])

                    # Add correct param grid key names
                    param_grid = {f"classifier__{key}": value for key, value in config["params"].items()}

                    grid = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        cv=5,
                        scoring='f1',
                        n_jobs=-1
                    )

                    grid.fit(X_train, y_train)
                    best_pipeline = grid.best_estimator_
                    # grid.fit(X_train, y_train)

                    # best_model = grid.best_estimator_

                    # pipeline = Pipeline([
                    #     ('classifier', best_model)
                    # ])
                    # pipeline.fit(X_train, y_train)

                    y_train_pred = best_pipeline.predict(X_train)
                    y_test_pred = best_pipeline.predict(X_test)

                    prob_df[f"{model_name}_tuned_prob"] = best_pipeline.predict_proba(X_test)[:, 1]

                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)

                    class_report = classification_report(y_test, y_test_pred, output_dict=True)
                    f1 = class_report['1'].get('f1-score', 0)
                    precision = class_report['1'].get('precision', 0)
                    recall = class_report['1'].get('recall', 0)
                    roc_auc = self.calculate_roc_auc(best_pipeline, X_test, y_test)

                    lift_score = self.plot_lift_gain(model_name, best_pipeline, X_test, y_test)
                    overfit_warning = get_overfit_warning(model_name, train_acc, test_acc, self.config.overfit_underfit_threshold)
                    lift_status = get_lift_status(model_name, lift_score)

                    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='f1')

                    cm = confusion_matrix(y_test, y_test_pred)
                    self.save_all_model_outputs(model_name, best_pipeline, cm, roc_auc, X_test, y_test)

                    # mlflow.log_params(grid.best_params_)
                    for k, v in grid.best_params_.items():
                        mlflow.log_param(k, v)
                    mlflow.log_metric("Train Accuracy", train_acc)
                    mlflow.log_metric("Test Accuracy", test_acc)
                    mlflow.log_metric("F1 Score", f1)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("ROC AUC", roc_auc)
                    mlflow.log_metric("Lift Score", lift_score)
                    mlflow.log_metric("CV F1 Mean", cv_scores.mean())
                    mlflow.log_metric("CV F1 Std", cv_scores.std())
                    mlflow.log_param("Lift Status", lift_status)
                    mlflow.log_param("Overfit/Underfit Warning", overfit_warning)

                    signature = infer_signature(X_test, y_test_pred)
                    mlflow.sklearn.log_model(
                        sk_model=best_pipeline,
                        name="model",
                        input_example=X_test.iloc[:5],
                        signature=signature
                    )

                    report.append({
                        "Model": model_name,
                        "Best Params": json.dumps(grid.best_params_),
                        "Train Accuracy": train_acc,
                        "Test Accuracy": test_acc,
                        "F1": f1,
                        "Precision": precision,
                        "Recall": recall,
                        "ROC AUC": roc_auc,
                        "Lift Score": lift_score,
                        "Overfit Warning": overfit_warning,
                        "Lift Status": lift_status,
                        "CV Mean": cv_scores.mean(),
                        "CV Std": cv_scores.std()
                    })

                    trained_pipelines[model_name] = best_pipeline

                    feature_df = self.plot_feature_importance(model=best_pipeline.named_steps['classifier'], X=X_test, model_name=model_name)
                    if feature_df is not None:
                        feature_dfs.append(feature_df)

            # Save all feature importances
            if feature_dfs:
                all_feature_df = pd.concat(feature_dfs)
                important_feat_csv_path = Path("artifacts/model_trained/metrics/top_features_tuned.csv")
                important_feat_csv_path.parent.mkdir(parents=True, exist_ok=True)
                all_feature_df.to_csv(important_feat_csv_path, index=False)
                logging.info(f"📌 Top N feature importance saved at {important_feat_csv_path}")

            # Save final predicted probabilities
            final_prob_csv = Path("artifacts/model_trained/metrics/test_predicted_probabilities_tuned.csv")
            os.makedirs(final_prob_csv.parent, exist_ok=True)
            prob_df.to_csv(final_prob_csv, index=False)
            logging.info(f"✅ All tuned model probabilities saved at {final_prob_csv}")

            # Save report
            report_df = pd.DataFrame(report)
            report_path = Path("artifacts/model_trained/metrics/tuned_model_comparison.csv")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_df.to_csv(report_path, index=False)
            logging.info(f"📊 Tuned model comparison report saved: {report_path}")
            self.save_combined_excel(
                report_df=report_df,
                output_excel_path="artifacts/model_trained/metrics/tuned_model_outputs.xlsx",
                top_features_csv="artifacts/model_trained/metrics/top_features_tuned.csv",
                predicted_prob_csv="artifacts/model_trained/metrics/test_predicted_probabilities_tuned.csv"
            )

            # Select best model
            best_model_row = report_df.sort_values(by="F1", ascending=False).iloc[0]
            best_model_name = best_model_row["Model"]
            best_pipeline = trained_pipelines[best_model_name]

            trained_model_filepath = os.path.join(self.config.trained_model_dir, f"{best_model_name}_tuned_model.pkl")
            os.makedirs(os.path.dirname(trained_model_filepath), exist_ok=True)
            joblib.dump(best_pipeline, trained_model_filepath)
            logging.info(f"\n ✅ Best tuned model is '{best_model_name}' saved at {trained_model_filepath}\n")

            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_filepath=trained_model_filepath,
                train_metric_filepath=self.config.train_metric_filepath,
                test_metric_filepath=self.config.test_metric_filepath,
                trained_conf_matrix=self.config.trained_conf_matrix,
                trained_roc_plot=self.config.trained_roc_plot,
                trained_shap_dir=self.config.trained_shap_dir,
                expected_accuracy=self.config.expected_accuracy,
                overfit_underfit_threshold=self.config.overfit_underfit_threshold,
                model_name=best_model_name,
                train_accuracy=best_model_row['Train Accuracy'],
                test_accuracy=best_model_row['Test Accuracy'],
                f1_score=best_model_row['F1'],
                precision=best_model_row['Precision'],
                recall=best_model_row['Recall'],
                roc_auc=best_model_row['ROC AUC']
            )

            return trained_pipelines, report_df, model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)


    # def train_tuned_models(self):
    #     logging.info('🔍 Starting Hyperparameter Tuning...')
    #     try:
    #         # Load data
    #         preprocessor = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)
    #         train_df = pd.read_csv(self.data_transformation_artifact.transformed_train_df_filepath)
    #         test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_df_filepath)

    #         X_train = train_df.iloc[:, :-1]
    #         y_train = train_df.iloc[:, -1]
    #         X_test = test_df.iloc[:, :-1]
    #         y_test = test_df.iloc[:, -1]

    #         tuned_models = {
    #             "RandomForest": {
    #                 "model": RandomForestClassifier(),
    #                 "params": {
    #                     "n_estimators": [50, 100],
    #                     "max_depth": [5, 10]
    #                 }
    #             },
    #             "DecisionTree": {
    #                 "model": DecisionTreeClassifier(),
    #                 "params": {
    #                     "max_depth": [3, 5, 10]
    #                 }
    #             },
    #             "LogisticRegression": {
    #                 "model": LogisticRegression(max_iter=1000),
    #                 "params": {
    #                     "C": [0.01, 0.1, 1, 10]
    #                 }
    #             },
    #             "XGBoost": {
    #                 "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    #                 "params": {
    #                     "n_estimators": [50, 100],
    #                     "max_depth": [3, 6]
    #                 }
    #             }
    #         }

    #         report = []

    #         for model_name, config in tuned_models.items():
    #             with mlflow.start_run(run_name=f"Tuned-{model_name}"):
    #                 grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='f1', n_jobs=-1)
    #                 grid.fit(X_train, y_train)

    #                 best_model = grid.best_estimator_
    #                 y_pred = best_model.predict(X_test)

    #                 test_acc = accuracy_score(y_test, y_pred)
    #                 class_report = classification_report(y_test, y_pred, output_dict=True)
    #                 f1 = class_report['1'].get('f1-score', 0)
    #                 precision = class_report['1'].get('precision', 0)
    #                 recall = class_report['1'].get('recall', 0)
    #                 roc_auc = self.calculate_roc_auc(best_model, X_test, y_test)

    #                 lift_score = self.plot_lift_gain(model_name, best_model, X_test, y_test)

    #                 mlflow.log_params(grid.best_params_)
    #                 mlflow.log_metric("Test Accuracy", test_acc)
    #                 mlflow.log_metric("F1 Score", f1)
    #                 mlflow.log_metric("Precision", precision)
    #                 mlflow.log_metric("Recall", recall)
    #                 mlflow.log_metric("ROC AUC", roc_auc)
    #                 mlflow.log_metric("Lift Score", lift_score)

    #                 mlflow.sklearn.log_model(best_model, artifact_path="model")

    #                 report.append({
    #                     "Model": model_name,
    #                     "Best Params": json.dumps(grid.best_params_),
    #                     "Test Accuracy": test_acc,
    #                     "F1": f1,
    #                     "Precision": precision,
    #                     "Recall": recall,
    #                     "ROC AUC": roc_auc,
    #                     "Lift Score": lift_score
    #                 })

    #         report_df = pd.DataFrame(report)
    #         report_path = Path("artifacts/model_trained/metrics/tuned_model_comparison.csv")
    #         report_df.to_csv(report_path, index=False)
    #         logging.info(f"📊 Tuned model report saved: {report_path}")

    #     except Exception as e:
    #         raise CustomException(e, sys)

    def calculate_roc_auc(self, pipeline, X_test, y_test):
        try:
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                probs = pipeline.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                return auc(fpr, tpr)
        except Exception:
            pass
        return np.nan

    def save_all_model_outputs(self, model_name, pipeline, cm, roc_auc, X_test, y_test):
        self.plot_confusion_matrix(model_name, cm)
        self.plot_roc_curve(model_name, pipeline, X_test, y_test, roc_auc)
        self.plot_feature_importance(model=pipeline.named_steps['classifier'], X=X_test, model_name=model_name)
        self.save_model(model_name, pipeline)

    def plot_class_distribution(y_train, title="Class Distribution Before SMOTETomek"):
        class_counts = Counter(y_train)
        labels = [f"Class {key}" for key in class_counts.keys()]
        sizes = list(class_counts.values())
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        plt.title(title)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()


    def plot_confusion_matrix(self, model_name, cm):
        self.config.trained_conf_matrix.mkdir(parents=True, exist_ok=True)
        img_path = self.config.trained_conf_matrix / f"{model_name}_conf_matrix.png"
        csv_path = self.config.trained_conf_matrix / f"{model_name}_conf_matrix.csv"

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        pd.DataFrame(cm).to_csv(csv_path)
        logging.info(f"✅ Saved confusion matrix: {img_path}")

    def plot_roc_curve(self, model_name, pipeline, X_test, y_test, roc_auc):
        if np.isnan(roc_auc):
            return

        fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
        path = self.config.trained_roc_plot / f"{model_name}_roc_curve.png"
        self.config.trained_roc_plot.mkdir(parents=True, exist_ok=True)

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        logging.info(f"✅ Saved ROC curve: {path}")



    def plot_lift_gain(self, model_name, pipeline, X_test, y_test):
        # Step 1: Get predicted probabilities for the positive class (churn = 1)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Step 2: Create a DataFrame with true values and predicted probabilities
        df = pd.DataFrame({'y_true': y_test, 'y_score': y_proba})
        
        # Step 3: Sort values by predicted probabilities
        df = df.sort_values('y_score', ascending=False)
        
        # Step 4: Group the sorted values into deciles (10 equal groups)
        df['decile'] = pd.qcut(df['y_score'], 10, labels=False, duplicates='drop')
        
        # Step 5: Calculate the Lift Table
        lift_table = df.groupby('decile').agg(
            total=('y_true', 'count'),
            events=('y_true', 'sum')
        ).sort_index(ascending=False).reset_index()

        # Step 6: Calculate cumulative values for events and total
        lift_table['cumulative_events'] = lift_table['events'].cumsum()
        lift_table['cumulative_total'] = lift_table['total'].cumsum()

        # Step 7: Calculate the Gain and Cumulative Gain
        lift_table['gain'] = lift_table['cumulative_events'] / lift_table['events'].sum()
        lift_table['cumulative_perc'] = lift_table['cumulative_total'] / lift_table['total'].sum()

        # Step 8: Calculate Lift
        lift_table['lift'] = lift_table['gain'] / lift_table['cumulative_perc']
        
        # Step 9: Get the lift score at the top decile
        top_decile_lift = lift_table.iloc[0]['lift']
        logging.info(f"Top Decile Lift for {model_name}: {top_decile_lift:4f}")

        path_csv = self.config.lift_gain_csv_path / f"{model_name}_lift_gain.csv"

        # Save the lift table as CSV for further inspection
        lift_table.to_csv(path_csv, index=False)
        # logging.info(f"✅ Saved Model Lift: {path_csv}")
        logging.info(f"✅ Saved Model Lift")

        # Return the top decile lift score as a rounded value
        return round(top_decile_lift, 3)
        

    def generate_conclusion(self, model_name, overfit, below_expected, lift_score, f1, roc_auc):
        issues = []
        if below_expected:
            issues.append("❌ Test accuracy below expected")
        if overfit:
            issues.append("⚠️ Overfitting detected")
        if lift_score < 1.2:
            issues.append("🔻 Low lift")
        if f1 < 0.5:
            issues.append("🟡 F1 score low")
        if roc_auc < 0.6:
            issues.append("🔸 ROC AUC underperforming")

        if not issues:
            return f"✅ {model_name} is strong, well-generalized, and effective."

        return f"{model_name} Summary: " + " | ".join(issues)


    def save_model(self, model_name, pipeline):
        path = self.config.trained_model_filepath.parent / f"{model_name}.pkl"
        self.config.trained_model_filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path)
        # logging.info(f"✅ Model saved: {path}")
    
    
    def save_combined_excel(
        self,
        report_df: pd.DataFrame,
        output_excel_path: str = "artifacts/model_trained/metrics/baseline_model_outputs.xlsx",
        top_features_csv: str = "artifacts/model_trained/metrics/top_features.csv",
        predicted_prob_csv: str = "artifacts/model_trained/metrics/test_predicted_probabilities.csv"
    ):
        """
        Saves a combined Excel file with model comparison report, top features, and predicted probabilities.
        Works for both baseline and tuned models by customizing input/output paths.

        Args:
            report_df (pd.DataFrame): The main report DataFrame (model comparison).
            output_excel_path (str): Path to the Excel file to be saved.
            top_features_csv (str): Path to the top features CSV.
            predicted_prob_csv (str): Path to the test probabilities CSV.
        """
        try:
            logging.info("📒 Creating combined Excel workbook...")

            # Read additional sheets if they exist
            top_feat_df = pd.read_csv(top_features_csv) if Path(top_features_csv).exists() else pd.DataFrame()
            prob_df = pd.read_csv(predicted_prob_csv) if Path(predicted_prob_csv).exists() else pd.DataFrame()

            # Save all sheets into one Excel file
            with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
                report_df.to_excel(writer, sheet_name="Model_Comparison", index=False)
                if not top_feat_df.empty:
                    top_feat_df.to_excel(writer, sheet_name="Top_Features", index=False)
                if not prob_df.empty:
                    prob_df.to_excel(writer, sheet_name="Test_Probabilities", index=False)

            logging.info(f"✅ Combined Excel Saved: {output_excel_path}")

        except Exception as e:
            logging.error(f"❌ Failed to save combined Excel workbook: {e}")
            raise CustomException(e, sys)



    def plot_feature_importance(self, model, X, model_name, top_n=10):
        feature_df = None
        plt.figure(figsize=(10, 6))
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = X.columns
            indices = np.argsort(importances)[::-1]
            top_features = features[indices][:top_n]
            top_importances = importances[indices][:top_n]

            plt.barh(top_features[::-1], top_importances[::-1])
            feature_df = pd.DataFrame({
                "Model": model_name,
                "Feature": top_features,
                "Importance": top_importances,
                "Rank": range(1, len(top_features)+1)
            })

        elif isinstance(model, LogisticRegression):
            coef = np.abs(model.coef_[0])
            features = X.columns
            indices = np.argsort(coef)[::-1]
            top_features = features[indices][:top_n]
            top_importances = coef[indices][:top_n]

            plt.barh(top_features[::-1], top_importances[::-1])
            feature_df = pd.DataFrame({
                "Model": model_name,
                "Feature": top_features,
                "Importance": top_importances,
                "Rank": range(1, len(top_features)+1)
            })
        else:
            logging.warning(f"⚠️ Feature importance not available for {model_name}")
            plt.close()
            return None  # Skip unknown model types

        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Features - {model_name}")
        plt.gca().invert_yaxis()
        os.makedirs(self.config.important_feat_plot, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.important_feat_plot, f"{model_name}_feature_importance.png"))
        plt.close()
        logging.info(f"✅ Feature importance plot saved for {model_name}\n")
        return feature_df




        
if __name__ == "__main__":
    # Step 1: Ingest the data
    ingestion_config = DataIngestionConfig()
    ingestion = DataIngestion(ingestion_config)
    ingestion_artifact = ingestion.initiate_data_ingestion()
    logging.info(f"Data Ingestion Artifact: {ingestion_artifact}\n")

    # Step 2: Validate the data using the artifact from ingestion
    validation_config = DataValidationConfig()
    validation = DataValidation(config=validation_config, data_ingestion_artifact=ingestion_artifact)
    validation_artifact = validation.initiate_data_validation()
    logging.info(f"Data Validation Artifact: {validation_artifact}\n")

    # Step 3: Transform the data using the artifact from Data VALIDATION
    transformation_config = DataTransformationConfig()
    transformation = DataTransformation(config=transformation_config, data_valid_artifact=validation_artifact)
    transformation_artifact = transformation.initiate_data_transformation()
    logging.info(f"Data Transformation Artifact: {transformation_artifact}\n")

    # Step 4: Training model data using the artifact from Data TRANSFORMATION
    mlflow.set_tracking_uri("https://dagshub.com/stat.data247/customer-churn-ml.git")
    mlflow.set_experiment("ChurnPredictionModels")
    # model_trainer_config = ModelTrainerConfig()
    # model_trainer = ModelTrainer(config=model_trainer_config, data_transformation_artifact=transformation_artifact)
    # model_trainer_artifact = model_trainer.train_baseline_models()
    # logging.info(f"Model Trainer Artifact: {model_trainer_artifact}\n")

    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(config=model_trainer_config, data_transformation_artifact=transformation_artifact)

    # 🚀 Train Baseline Models
    baseline_pipelines, baseline_report_df, baseline_artifact = model_trainer.train_baseline_models()
    logging.info(f"📦 Baseline Model Trainer Artifact: {baseline_artifact}\n")

    # 🔍 Train Tuned Models
    tuned_pipelines, tuned_report_df, tuned_artifact = model_trainer.train_tuned_models()
    logging.info(f"⚙️ Tuned Model Trainer Artifact: {tuned_artifact}\n")

    # (Optional) Save a combined model comparison
    combined_report_df = pd.concat([
        baseline_report_df.assign(TrainingType="Baseline"),
        tuned_report_df.assign(TrainingType="Tuned")
    ])

    combined_excel_path = "artifacts/model_trained/metrics/full_model_comparison.xlsx"
    model_trainer.save_combined_excel(
        report_df=combined_report_df,
        output_excel_path=combined_excel_path,
        top_features_csv="artifacts/model_trained/metrics/top_features_tuned.csv",
        predicted_prob_csv="artifacts/model_trained/metrics/test_predicted_probabilities_tuned.csv"
    )

    logging.info("✅ All model training completed successfully.")