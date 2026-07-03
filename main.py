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
from Churn_Pred.components.model_Pusher import ModelTrainer,ModelTrainerArtifact, ModelTrainerConfig
from Churn_Pred.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact
from Churn_Pred.utils import get_lift_status, get_overfit_warning
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from collections import Counter
import time
from datetime import datetime

if __name__ == "__main__":
    try:
        pipeline_start = time.now()
        logging.info("=" * 80)
        logging.info("STARTING READMISSION PREDICTION PIPELINE")
        logging.info("=" * 80)

        # ------------------------------------------------------------------
        # Step 1 : Data Ingestion
        # ------------------------------------------------------------------
        ingestion_config = DataIngestionConfig()
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion Artifact:\n{ingestion_artifact}")

        # ------------------------------------------------------------------
        # Step 2 : Data Validation
        # ------------------------------------------------------------------
        validation_config = DataValidationConfig()
        validation = DataValidation(config=validation_config, data_ingestion_artifact=ingestion_artifact)
        validation_artifact = validation.initiate_data_validation()
        logging.info(f"Data Validation Artifact:\n{validation_artifact}")

        # ------------------------------------------------------------------
        # Step 3 : Data Transformation
        # ------------------------------------------------------------------
        transformation_config = DataTransformationConfig()
        transformation = DataTransformation(config=transformation_config, data_valid_artifact=validation_artifact)
        transformation_artifact = transformation.initiate_data_transformation()
        logging.info(f"Data Transformation Artifact:\n{transformation_artifact}")

        logging.info(f"Training Dataset : {transformation_artifact.transformed_train_file_path}")
        logging.info(f"Testing Dataset : {transformation_artifact.transformed_test_file_path}")
        # ------------------------------------------------------------------
        # Step 4 : Model Training
        # ------------------------------------------------------------------
        trainer_config = ModelTrainerConfig()
        trainer = ModelTrainer(config=trainer_config, data_transformation_artifact=transformation_artifact)
        logging.info("=" * 80)
        logging.info("TRAINING BASELINE MODELS")
        logging.info("=" * 80)
        (
            baseline_models,
            baseline_report,
            baseline_artifact
        ) = trainer.train_baseline_models()
        logging.info("Baseline training completed.")
        logging.info("=" * 80)
        logging.info("TRAINING TUNED MODELS")
        logging.info("=" * 80)

        (
            tuned_models,
            tuned_report,
            tuned_artifact
        ) = trainer.train_tuned_models()
        logging.info("Hyperparameter tuning completed.")

        best_model = tuned_report.sort_values(by="ROC_AUC", ascending=False).iloc[0]
        logging.info(
            f"Best tuned model: {best_model['Model']} "
            f"(ROC-AUC={best_model['ROC_AUC']:.4f})")
        
        best_model.to_frame().T.to_csv("artifacts/model_trained/metrics/best_model.csv", index=False)

        pipeline_metadata = {
            "best_model": best_model["Model"],
            "roc_auc": float(best_model["ROC_AUC"]),
            "execution_time": time.perf_counter() - pipeline_start,
            "timestamp": datetime.now().isoformat()
        }

        with open("artifacts/model_trained/pipeline_metadata.json", "w") as f:
            json.dump(pipeline_metadata,f, indent=4)
        # ------------------------------------------------------------------
        # Step 5 : Model Comparison
        # ------------------------------------------------------------------
        combined_report = pd.concat(
            [
                baseline_report.assign(TrainingType="Baseline"),
                tuned_report.assign(TrainingType="Tuned")
            ],
            ignore_index=True
        )
        logging.info("\n%s", combined_report.round(4))

        trainer.save_combined_excel(
            report_df=combined_report,
            output_excel_path="artifacts/model_trained/metrics/full_model_comparison.xlsx",
            top_features_csv="artifacts/model_trained/metrics/top_features_tuned.csv",
            predicted_prob_csv="artifacts/model_trained/metrics/test_predicted_probabilities_tuned.csv"
        )
        logging.info("=" * 80)
        logging.info("🎉 READMISSION PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)
        logging.info(f"🏆 Best Model : {best_model['Model']}")
        logging.info(f"📈 ROC-AUC    : {best_model['ROC_AUC']:.4f}")
        logging.info("=" * 80)

        logging.info("Artifacts saved to:")
        logging.info("  artifacts/model_trained/")
        logging.info("  artifacts/model_trained/shap/")
        logging.info("  artifacts/model_trained/images/")
        logging.info("  artifacts/model_trained/metrics/")
    except Exception as e:
        logging.exception("=" * 80)
        logging.exception("PIPELINE FAILED")
        logging.exception("=" * 80)
        raise