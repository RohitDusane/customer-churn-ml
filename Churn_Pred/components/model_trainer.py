# model_trainer.py

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
from sklearn.model_selection import cross_val_score
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
from Churn_Pred.utils import print_lift_status, print_overfit_warning


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
    expected_accuracy: float = 0.6
    overfit_underfit_threshold: float = 0.05


# ====== MODEL TRAINER CLASS ======
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact

    def save_model_trainer_artifact(self, artifact: ModelTrainerArtifact, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(artifact.__dict__, f, indent=4)
        print(f"ModelTrainerArtifact saved at {filepath}")

    def train_models(self):
        logging.info(' ✅ Initiating Model Training...')
        try:
            preprocessor = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)
            # Load the transformed training data
            # train_arr = np.load(self.data_transformation_artifact.transformed_train_arr_filepath)
            # test_arr = np.load(self.data_transformation_artifact.transformed_test_arr_filepath)

            train_df = pd.read_csv(self.data_transformation_artifact.transformed_train_df_filepath)
            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_df_filepath)
            logging.info('Loading preprocessor, train and test datasets...')
            
            # Split features and target
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            
            print(f"Loading preprocessed data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
            

            # model_trainer.train_models(X_train, y_train, X_test, y_test)

            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            print(f"Loading preprocessed data: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
            

            models = {
                'RandomForest': RandomForestClassifier(verbose=1),
                'DecisionTree': DecisionTreeClassifier(),
                'LogisticRegression': LogisticRegression(),
                'Xgb': XGBClassifier()
            }

            report = []
            trained_pipelines = {}

            for model_name, model in models.items():
                logging.info(f"🔧 Training {model_name}")

                pipe = Pipeline([
                    ('classifier', model)
                ])

                pipe.fit(X_train, y_train)

                # Model Predictions
                y_train_pred = pipe.predict(X_train)
                y_test_pred = pipe.predict(X_test)

                # Model Accuracy
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)

                # Check Test Accuracy meets Expected Accuracy criteria or not
                if test_acc < self.config.expected_accuracy:
                    logging.warning(f"{model_name} below BAR threshold! Test accuracy = {test_acc}")

                # Check wheter overfit/underfit and its levels
                self.print_overfit_warning(model_name, train_acc, test_acc, threshold=self.config.overfit_underfit_threshold)
                # logging.info(f"SHAP values for  {model_name}: ")
                # self.plot_shap_summary(model_name=model_name, model=pipe, X_test=X_test)

                class_report = classification_report(y_test, y_test_pred, output_dict=True)
                f1 = class_report['1'].get('f1-score', 0)
                precision = class_report['1'].get('precision', 0)
                recall = class_report['1'].get('recall', 0)
                roc_auc = self.calculate_roc_auc(pipe, X_test, y_test)

                cm = confusion_matrix(y_test, y_test_pred)
                self.save_all_model_outputs(model_name, pipe, cm, roc_auc, X_test, y_test)

                cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')

                # Lift score
                lift_score = self.calculate_lift_score(pipe, X_test, y_test)
                print_lift_status(model_name, lift_score)

                report.append({
                    "Model": model_name,
                    "Train Accuracy": train_acc,
                    "Test Accuracy": test_acc,
                    "F1": f1,
                    "Precision": precision,
                    "Recall": recall,
                    "ROC AUC": roc_auc,
                    "CV Mean": cv_scores.mean(),
                    "CV Std": cv_scores.std()
                })

                # Keep pipeline for later use (to save best)
                trained_pipelines[model_name] = pipe

                

            report_df = pd.DataFrame(report)
            self.save_comparison_report(report_df)

            # --- Save best model and artifact ---
            best_model_row = report_df.loc[report_df['F1'].idxmax()]
            best_model_name = best_model_row['Model']
            best_pipeline = trained_pipelines[best_model_name]

            # Define model save path
            trained_model_filepath = os.path.join(self.config.trained_model_dir, f"{best_model_name}_model.pkl")

            # Ensure the directory exists
            model_dir = os.path.dirname(trained_model_filepath)
            os.makedirs(model_dir, exist_ok=True)

            # Save best model pipeline
            joblib.dump(best_pipeline, trained_model_filepath)
            print(f"Best model '{best_model_name}' saved at {trained_model_filepath}")

            

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

            # Save artifact as JSON
            # artifact_path = os.path.join(self.config.artifact_dir, "model_trainer_artifact.json")
            # self.save_model_trainer_artifact(model_trainer_artifact, artifact_path)

            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
    

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
        self.plot_lift_gain(model_name, pipeline, X_test, y_test)
        # self.plot_shap_summary(model_name=model_name, model=pipeline, X_test=X_test)
        self.save_model(model_name, pipeline)

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
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        df = pd.DataFrame({'y_true': y_test, 'y_score': y_proba})
        df = df.sort_values('y_score', ascending=False)
        df['decile'] = pd.qcut(df['y_score'], 10, labels=False, duplicates='drop')

        lift_table = df.groupby('decile').agg(
            total=('y_true', 'count'),
            events=('y_true', 'sum')
        ).sort_index(ascending=False).reset_index()

        lift_table['cumulative_events'] = lift_table['events'].cumsum()
        lift_table['cumulative_total'] = lift_table['total'].cumsum()
        lift_table['gain'] = lift_table['cumulative_events'] / lift_table['events'].sum()
        lift_table['cumulative_perc'] = lift_table['cumulative_total'] / lift_table['total'].sum()
        lift_table['lift'] = lift_table['gain'] / lift_table['cumulative_perc']

        top_decile_lift = lift_table.iloc[0]['lift']
        return round(top_decile_lift, 3)

        # # Plot
        # plt.plot(lift_table['cumulative_perc'], lift_table['gain'], label='Gain')
        # plt.plot(lift_table['cumulative_perc'], lift_table['lift'], label='Lift')
        # plt.axhline(1, color='gray', linestyle='--')
        # plt.xlabel("Cumulative %")
        # plt.ylabel("Gain / Lift")
        # plt.legend()
        # plt.title(f"Lift/Gain - {model_name}")
        # plt.tight_layout()

        # path_img = self.config.trained_roc_plot / f"{model_name}_lift_gain.png"
        # path_csv = self.config.trained_roc_plot / f"{model_name}_lift_gain.csv"
        # plt.savefig(path_img)
        # plt.close()




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
        logging.info(f"✅ Model saved: {path}")

    def save_comparison_report(self, report_df: pd.DataFrame):
        path = Path("artifacts/model_trained/metrics/model_comparison.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(path, index=False)
        logging.info(f"📊 Model comparison report saved: {path.resolve()}")
        print(report_df)
    
    def save_comparison_report(self, report_df: pd.DataFrame):
        try:
            path = Path("artifacts/model_trained/metrics/model_comparison.xlsx")
            path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent folders exist

            report_df.to_excel(path, index=False)
            
            logging.info(f"📊 Model comparison report saved at: {path.resolve()}")
            print("\n📊 Model Comparison Report:\n", report_df)
        except Exception as e:
            logging.error(f"❌ Failed to save model comparison report: {e}")
            raise CustomException(e,sys)
    
    # def plot_shap_summary(self, model_name, model, X_test):
    #     try:
    #         self.config.trained_shap_dir.mkdir(parents=True, exist_ok=True)

    #         # ✅ Load the preprocessor object from file
    #         # logging.info(f"📦 Loading preprocessing pipeline OBJECT from: {self.data_transformation_artifact.transformed_preprocessor_obj_filepath}")
    #         # preprocessing_pipeline = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)

    #         logging.info(f"🔍 Transforming X_test using preprocessing pipeline...")
    #         X_transformed = X_test

    #         # Convert to dense array if sparse
    #         if hasattr(X_transformed, "toarray"):
    #             X_transformed = X_transformed.toarray()

    #         logging.info(f"✅ X_test transformed: shape={X_transformed.shape}")

    #         # ✅ Get feature names
    #         # if hasattr(preprocessing_pipeline, 'get_feature_names_out'):
    #         #     feature_names = preprocessing_pipeline.get_feature_names_out()
    #         # else:
    #         #     feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    #         # ✅ Create SHAP explainer
    #         logging.info("🔍 Creating SHAP explainer...")
    #         if hasattr(model, "predict_proba"):
    #             explainer = shap.Explainer(model.predict_proba, X_transformed)
    #         else:
    #             explainer = shap.Explainer(model, X_transformed)

    #         # ✅ Compute SHAP values
    #         logging.info("📊 Calculating SHAP values...")
    #         shap_values = explainer(X_transformed)

    #         # ✅ Save SHAP values as CSV
    #         shap_df = pd.DataFrame(shap_values.values, columns=shap_values.index)
    #         shap_df.index = X_test.index if hasattr(X_test, 'index') else range(len(X_test))

    #         shap_csv_path = self.config.trained_shap_dir / f"shap_values_{model_name}.csv"
    #         shap_df.to_csv(shap_csv_path)
    #         logging.info(f"✅ SHAP values saved: {shap_csv_path}")

    #         # Optional: Check for empty SHAP values
    #         if hasattr(shap_values, "values") and np.all(shap_values.values == 0):
    #             logging.warning(f"⚠️ All SHAP values are zero for {model_name}")
    #             return

    #         # ✅ Plot summary
    #         logging.info("📊 Creating SHAP summary plot...")
    #         shap.summary_plot(shap_values, features=X_transformed, feature_names=feature_names, show=False)

    #         shap_img_path = self.config.trained_shap_dir / f"shap_{model_name}.png"
    #         plt.title(f"SHAP Summary: {model_name}")
    #         plt.savefig(shap_img_path)
    #         plt.close()

    #         logging.info(f"✅ SHAP plot saved: {shap_img_path}")

    #     except Exception as e:
    #         logging.warning(f"⚠️ SHAP generation failed for {model_name}: {e}")




    




# if __name__ == "__main__":
#     # Step 1: Ingest the data
#     ingestion_config = DataIngestionConfig()
#     ingestion = DataIngestion(ingestion_config)
#     ingestion_artifact = ingestion.initiate_data_ingestion()
#     logging.info(f"Data Ingestion Artifact: {ingestion_artifact}")

#     # Step 2: Validate the data using the artifact from ingestion
#     validation_config = DataValidationConfig()
#     validation = DataValidation(config=validation_config, data_ingestion_artifact=ingestion_artifact)
#     validation_artifact = validation.initiate_data_validation()
#     logging.info(f"Data Validation Artifact: {validation_artifact}")

#     # Step 3: Transform the data using the artifact from Data VALIDATION
#     transformation_config = DataTransformationConfig()
#     transformation = DataTransformation(config=transformation_config, data_valid_artifact=validation_artifact)
#     transformation_artifact = transformation.initiate_data_transformation()
#     logging.info(f"Data Transformation Artifact: {transformation_artifact}")

#     # Step 4: Training model data using the artifact from Data TRANSFORMATION
#     model_trainer_config = ModelTrainerConfig()
#     model_trainer = ModelTrainer(config=model_trainer_config, data_transformation_artifact=transformation_artifact)
#     model_trainer_artifact = model_trainer.train_models()
#     logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")



































# import os,sys
# from pathlib import Path
# from Churn_Pred.exception.exception import CustomException
# from Churn_Pred.logger.log import logging
# from dataclasses import dataclass

# from Churn_Pred.components.data_ingestion import DataIngestionConfig, DataIngestion
# from Churn_Pred.components.data_validation import DataValidationConfig, DataValidation
# from Churn_Pred.components.data_transformation import DataTransformationConfig, DataTransformation
# from Churn_Pred.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact
# from Churn_Pred.utils import *

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# # from xgboost import XGBClassifier

# import shap



# from sklearn.metrics import roc_curve, confusion_matrix, auc, accuracy_score, classification_report


# # ------ Model Trainer Configuration ------
# @dataclass
# class ModelTrainerConfig:
#     trained_model_filepath: Path = Path('artifacts','model_trained',"models",'model.pkl')
#     train_metric_filepath: Path = Path('artifacts','model_trained',"metrics", 'train.npy')
#     test_metric_filepath : Path = Path('artifacts','model_trained', "metrics", 'test.npy')
#     trained_conf_matrix: Path = Path('artifacts','model_trained', "images")
#     trained_roc_plot: Path = Path('artifacts','model_trained', "images")
#     trained_shap_dir: Path = Path("artifacts", "model_trained", "shap")
#     expected_accuracy:float = 0.6
#     overfit_underfit_threshold:float = 0.05


# # ------ Model Trainer class ------
# class ModelTrainer:
#     def __init__(self,config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
#         try:
#             self.config = config
#             self.data_transformation_artifact = data_transformation_artifact
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     # def evaluate_models():
#     #     try:
#     #         pass
#     #     except Exception as e:
#     #         raise CustomException(e,sys)
        
#     def save_model_metrics(self, model_name, cm, roc_auc, model, X_test, y_test):
#         """Save confusion matrix, ROC curve, and other metrics."""
#         try:
#             # cm_path = os.path.join(self.config.trained_conf_matrix, f"{model_name}_conf_matrix.png")
#             # roc_path = os.path.join(self.config.trained_roc_plot, f"{model_name}_roc_curve.png")
#             # Ensure directory exists
#             self.config.trained_conf_matrix.mkdir(parents=True, exist_ok=True)
#             self.config.trained_roc_plot.mkdir(parents=True, exist_ok=True)

#             # File paths
#             cm_img_path = self.config.trained_conf_matrix / f"{model_name}_conf_matrix.png"
#             cm_csv_path = self.config.trained_conf_matrix / f"{model_name}_conf_matrix.csv"
#             roc_path = self.config.trained_roc_plot / f"{model_name}_roc_curve.png"
#             lift_gain_path = self.config.trained_roc_plot / f"{model_name}_lift_gain.png"
#             lift_gain_csv = self.config.trained_roc_plot / f"{model_name}_lift_gain.csv"
            
            
#             # Save Confusion Matrix
#             plt.figure(figsize=(6, 4))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#             plt.title(f"Confusion Matrix - {model_name}")
#             plt.xlabel("Predicted")
#             plt.ylabel("Actual")
#             plt.tight_layout()
#             plt.savefig(cm_img_path)
#             plt.close()

#             # -------- Save Confusion Matrix as CSV --------
#             cm_df = pd.DataFrame(cm, columns=['Predicted_0', 'Predicted_1'], index=['Actual_0', 'Actual_1'])
#             cm_df.to_csv(cm_csv_path)
#             print(f"✅ Saved Confusion Matrix: {cm_img_path}, {cm_csv_path}")

#             # Save ROC Curve
#             fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
#             plt.figure(figsize=(6, 4))
#             plt.plot(fpr, tpr, color='darkorange', lw=1.5, label=f'ROC curve (area = {roc_auc:.4f})')
#             plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title(f'ROC Curve - {model_name}')
#             plt.legend(loc='lower right')
#             plt.tight_layout()
#             plt.savefig(roc_path)
#             plt.close()

#             # -------- Lift/Gain Chart --------
#             y_proba = model.predict_proba(X_test)[:, 1]
#             df_lift = pd.DataFrame({'y_true': y_test, 'y_score': y_proba})
#             df_lift = df_lift.sort_values('y_score', ascending=False)
#             df_lift['decile'] = pd.qcut(df_lift['y_score'], 10, labels=False, duplicates='drop')

#             lift_table = df_lift.groupby('decile').agg(
#                 total=('y_true', 'count'),
#                 events=('y_true', 'sum')
#             ).sort_index(ascending=False).reset_index()

#             total_events = lift_table['events'].sum()
#             lift_table['cumulative_events'] = lift_table['events'].cumsum()
#             lift_table['cumulative_total'] = lift_table['total'].cumsum()
#             lift_table['gain'] = lift_table['cumulative_events'] / total_events
#             lift_table['cumulative_perc'] = lift_table['cumulative_total'] / lift_table['total'].sum()
#             lift_table['lift'] = lift_table['gain'] / lift_table['cumulative_perc']
#             lift_table['lift_percent'] = (lift_table['lift'] - 1) * 100

#             # --- Plot Lift & Gain ---
#             plt.figure(figsize=(6, 4))
#             plt.plot(lift_table['cumulative_perc'], lift_table['gain'], marker='o', label='Gain')
#             plt.plot(lift_table['cumulative_perc'], lift_table['lift'], marker='x', label='Lift')
#             plt.axhline(1, color='gray', linestyle='--', label='Baseline Lift')
#             plt.xlabel("Cumulative % of Samples")
#             plt.ylabel("Gain / Lift")
#             plt.title(f"Lift/Gain Chart - {model_name}")
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(lift_gain_path)
#             plt.close()

#             lift_table.to_csv(lift_gain_csv, index=False)
#             print(f"✅ Saved Lift/Gain Plot: {lift_gain_path}")
#             print(f"✅ Saved Lift/Gain CSV: {lift_gain_csv}")
#             base_rate = total_events / lift_table['total'].sum()
#             BAR = 0.5
#             top_decile_lift = lift_table.iloc[0]['lift']
#             print_lift_status(model_name, top_decile_lift, bar=BAR)
            


#             print(f"🚀 Top decile Lift over baseline: {top_decile_lift:.2f}x (i.e. {((top_decile_lift - 1) * 100):.1f}%)")
        
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def plot_shap_summary(self, model_name: str, pipe, X_test):
#         """
#         Generate and save SHAP summary plot for a model wrapped in a sklearn Pipeline.
#         """
#         try:
#             self.config.trained_shap_dir.mkdir(parents=True, exist_ok=True)

#             # Transform test data using pipeline
#             X_transformed = pipe.named_steps['preprocessing'].transform(X_test)

#             # Extract model
#             model = pipe.named_steps['classifier']

#             # Create SHAP explainer
#             explainer = shap.Explainer(model, X_transformed)
#             shap_values = explainer(X_transformed)

#             # Plot SHAP summary
#             shap.summary_plot(shap_values, X_transformed, show=False)
#             plot_path = self.config.trained_shap_dir / f"shap_{model_name.replace(' ', '_').lower()}.png"
#             plt.title(f"SHAP Summary: {model_name}")
#             plt.savefig(plot_path)
#             plt.close()

#             print(f"✅ SHAP plot saved to: {plot_path}")

#         except Exception as e:
#             logging.warning(f"❌ SHAP failed for {model_name}: {e}")
#             raise CustomException(e,sys)
        
#     def get_best_model(self, report_df: pd.DataFrame, metric: str = "F1") -> str:
#         try:
#             best_model = report_df.sort_values(by=metric, ascending=False).iloc[0]
#             print(f"\n🏆 Best model: {best_model['Model']} with {metric}: {best_model[metric]:.4f}")
#             return best_model['Model']
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def train_models(self, X_train, y_train, X_test, y_test):
#         try:
#             preprocessor = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)
#             models = {
#                 'RF': RandomForestClassifier(verbose=1),
#                 'DT': DecisionTreeClassifier(),
#                 'LogR': LogisticRegression()
#             }

#             report_list = []

#             for name, model in models.items():
#                 logging.info(f"\nTraining Model - {name}")

#                 pipe = Pipeline([
#                     ('preprocessing', preprocessor),
#                     ('classifier', model)
#                 ])

#                 # === Train ===
#                 pipe.fit(X_train, y_train)

#                 # === Predict ===
#                 y_train_pred = pipe.predict(X_train)
#                 y_test_pred = pipe.predict(X_test)

#                 # === Classification Report ===
#                 class_report = classification_report(y_test, y_test_pred, output_dict=True)

                
#                 train_acc = accuracy_score(y_train, y_train_pred)
#                 test_acc = accuracy_score(y_test, y_test_pred)
#                 f1 = class_report['1']['f1-score']
#                 precision = class_report['1']['precision']
#                 recall = class_report['1']['recall']
#                 print_overfit_warning(name, train_acc, test_acc, threshold=self.config.overfit_underfit_threshold)
                


#                 # === Cross-Validation ===
#                 cv_score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')

#                 # === Confusion Matrix ===
#                 cm = confusion_matrix(y_test, y_test_pred)

#                 # === ROC AUC ===
#                 if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
#                     y_proba = pipe.predict_proba(X_test)[:, 1]
#                     fpr, tpr, _ = roc_curve(y_test, y_proba)
#                     roc_auc = auc(fpr, tpr)
#                 else:
#                     roc_auc = np.nan

#                 # === Save Metrics & Plots ===
#                 self.save_model_metrics(name, cm, roc_auc, pipe, X_test, y_test)

#                 # === SHAP Plots ===
#                 self.plot_shap_summary(name, pipe, X_test)

#                 # === Save Model ===
#                 self.config.trained_model_filepath.parent.mkdir(parents=True, exist_ok=True)
#                 model_path = self.config.trained_model_filepath.parent / f"{name}.pkl"
#                 joblib.dump(pipe, model_path)

#                 # === Append Report ===
#                 report_list.append({
#                     "Model": name,
#                     "TRAIN Accuracy": train_acc,
#                     "TEST Accuracy": test_acc,
#                     "F1": f1,
#                     "Precision": precision,
#                     "Recall": recall,
#                     "CV Mean": cv_score.mean(),
#                     "CV Std": cv_score.std(),
#                     "ROC AUC": roc_auc
#                 })

#             # === Export Report ===
#             report_df = pd.DataFrame(report_list)
#             report_path = Path("artifacts", "model_trained", "metrics", "model_comparison.xlsx")
#             report_path.parent.mkdir(parents=True, exist_ok=True)
#             report_df.to_excel(report_path, index=False)

#             print("\n✅ Report saved to:", report_path)
#             print(report_df)

#         except Exception as e:
#             raise CustomException(e, sys)

        
#     def evaluate_models(self, model, X_train, X_test, y_train, y_test, model_name):
#         try: 
#             """Evaluate model performance on train and Test sets."""
#             y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
#             train_class_report = classification_report(y_train, y_train_pred, output_dict=True)
#             test_class_report = classification_report(y_test, y_test_pred, output_dict=True)

#             # Collect metrics
#             precision_train = train_class_report.get('1', {}).get('precision', 0.0)
#             recall_train = train_class_report.get('1', {}).get('recall', 0.0)
#             f1_train = train_class_report.get('1', {}).get('f1-score', 0.0)
#             precision_test = test_class_report.get('1', {}).get('precision', 0.0)
#             recall_test = test_class_report.get('1', {}).get('recall', 0.0)
#             f1_test = test_class_report.get('1', {}).get('f1-score', 0.0)

#             # Return performance dictionary
#             return {
#                 'Model': model_name,
#                 'Train Accuracy': accuracy_score(y_train, y_train_pred),
#                 'Test Accuracy': accuracy_score(y_test, y_test_pred),
#                 'Train Precision': round(precision_train, 4),
#                 'Test Precision': round(precision_test, 4),
#                 'Train Recall': round(recall_train, 4),
#                 'Test Recall': round(recall_test, 4),
#                 'Train F1-Score': round(f1_train, 4),  
#                 'Test F1-Score': round(f1_test, 4)
#             }
#         except Exception as e:
#             raise CustomException(e,sys)