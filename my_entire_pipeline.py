# ============================================================================================
# ============================================================================================
# =============== DATA_INGESTION =============================================================
# ============================================================================================
# ============================================================================================
import os
import sys
from pathlib import Path
from Churn_Pred.exception.exception import CustomException
from Churn_Pred.logger.log import logging
from Churn_Pred.utils import lowercase_columns
from Churn_Pred.entity.artifact_entity import DataIngestionArtifact

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# ----- Data Ingestion Configuration -------
@dataclass
class DataIngestionConfig:
    raw_data_path: Path = Path('artifacts/rawdata.csv')
    train_data_path: Path = Path('artifacts/ingested/train.csv')
    test_data_path: Path = Path('artifacts/ingested/test.csv')
    test_size: float = 0.2
    random_state: int = 24
    num_train_rows: int = 0
    num_test_rows: int = 0

# ----- Data Ingestion Class -------
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info('Starting Data Ingestion step')

        try:
            # Step 1: Ensure the raw data exists
            if not self.config.raw_data_path.exists():
                logging.error(f"Raw data file not found at {self.config.raw_data_path}")
                raise FileNotFoundError(f"Missing raw data file at {self.config.raw_data_path}")

            # Step 2: Read the raw data
            logging.info("Reading raw data...")
            df = pd.read_csv(self.config.raw_data_path)


            # Step 3: Drop unwanted columns early
            drop_columns = ['rownumber', 'customerid', 'surname']
            df.columns = df.columns.str.strip()
            df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
            logging.info(f"Dropped columns: {drop_columns}")

            # Step 4: Lowercase all column names
            df = lowercase_columns(df)
            logging.info(f"Dataset shape after cleaning: {df.shape}")

            # Step 5: Save the cleaned raw data back to the raw path
            self.config.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, encoding='utf-8')
            logging.info(f"Cleaned raw data saved at {self.config.raw_data_path}")

            # Step 6: Train-test split
            logging.info("Splitting data into train and test sets...")
            train_set, test_set = train_test_split(df, 
                                                   test_size=self.config.test_size,
                                                   random_state=self.config.random_state)

            # Ensure the directories for train and test data exist
            self.config.train_data_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.test_data_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the train and test data
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info(f"Train data saved at {self.config.train_data_path}")
            logging.info(f"Test data saved at {self.config.test_data_path}")

            # Return the artifact with data paths
            artifact = DataIngestionArtifact(
                train_data_path=self.config.train_data_path,
                test_data_path=self.config.test_data_path,
                num_train_rows=train_set.shape[0],
                num_test_rows=test_set.shape[0]
            )

            logging.info('Data ingestion step completed successfully.')

            return artifact

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)






# ============================================================================================
# ============================================================================================
# =============== DATA TRANSFORMATION ========================================================
# ============================================================================================
# ============================================================================================
import os,sys
from pathlib import Path
from Churn_Pred.exception.exception import CustomException
from Churn_Pred.logger.log import logging
from dataclasses import dataclass

from Churn_Pred.components.data_ingestion import DataIngestionConfig, DataIngestion
from Churn_Pred.components.data_validation import DataValidationConfig, DataValidation
from Churn_Pred.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact, DataValidationArtifact

import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------ Data Transformation Configuration ------
@dataclass
class DataTransformationConfig:
    # preprocessor_obj_filepath:Path = Path('artifacts','transformed','preprocessor.pkl')
    transformed_train_arr_filepath: Path = Path('artifacts','transformed','train.npy')
    transformed_test_arr_filepath: Path = Path('artifacts','transformed','test.npy')
    transformed_train_df_filepath: Path = Path('artifacts','transformed', 'df', 'train_df.csv')
    transformed_test_df_filepath: Path =  Path('artifacts','transformed', 'df', 'test_df.csv')
    transformed_preprocessor_obj_filepath : Path = Path('artifacts','transformed','preprocessor.pkl')
    target_columns:str = 'exited'


# ------ Data Transformation class ------
class DataTransformation:
    def __init__(
            self,
            config:DataTransformationConfig, 
            data_valid_artifact:DataValidationArtifact
        ):
        try:
            self.config = config
            self.data_valid_artifact = data_valid_artifact
        except Exception as e:
            raise CustomException(e,sys)

    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def validate_required_columns(df):
        expected_columns = ['creditscore','geography','gender','age','tenure','balance',
                            'numofproducts','hascrcard','isactivemember','estimatedsalary']
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True


    def get_data_transformer_object(self, df:pd.DataFrame):
        """
        Creates and returns a ColumnTransformer object for preprocessing.
        """
        try:
            # Explicitly drop target column before identifying feature types
            TARGET_COLUMN = self.config.target_columns
            df = df.drop(columns=[TARGET_COLUMN], errors='ignore')

            # Select numerical & categorical Cols
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ], verbose=True)

            logging.info('Categorical pipeline defined.')

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ], verbose=True)

            logging.info('Numerical pipeline defined.')

            preprocessor = ColumnTransformer(transformers=[
                ('cat', cat_pipeline, categorical_cols),
                ('num', num_pipeline, numeric_cols)
            ], remainder='drop', verbose=True)

            logging.info(f'Preprocessor created with numeric columns: {numeric_cols}')
            logging.info(f'Preprocessor created with categorical columns: {categorical_cols}')

            logging.info('Preprocessor (ColumnTransformer) constructed.')

            # save_path = self.config.transformed_preprocessor_obj_filepath
            # dir_path = os.path.dirname(save_path)
            # os.makedirs(dir_path, exist_ok=True)  # Ensures directory exists
            # joblib.dump(preprocessor, save_path)
            # joblib.dump(scaler, 'artifacts/models/scaler.joblib')

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)



    def initiate_data_transformation(self):
        logging.info('Initiating Data Transformation...')
        try:
            # Load validated train and test data
            logging.info("Loading ingested datasets...")
            train_data = self.data_valid_artifact.valid_train_file_path
            test_data = self.data_valid_artifact.valid_test_file_path

            train_df = DataTransformation.read_data(train_data)
            test_df = DataTransformation.read_data(test_data)

            logging.info("Obtaining Preprocessor object")
            preprocessor_obj = self.get_data_transformer_object(train_df)

            TARGET_COLUMN = self.config.target_columns
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # Validate required columns
            try:
                DataTransformation.validate_required_columns(X_train)
                DataTransformation.validate_required_columns(X_test)
            except Exception as e:
                raise CustomException(e,sys)

            # Fit and transform data
            logging.info("Fitting and transforming training data")
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            logging.info("Transforming test data using fitted preprocessor")
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Get feature names (optional, but recommended)
            try:
                feature_names = preprocessor_obj.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

            # Convert to DataFrames
            X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

            # Add target column back
            train_df_final = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
            test_df_final = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

            # Save transformed DataFrames as CSVs
            os.makedirs(os.path.dirname(self.config.transformed_train_df_filepath), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.transformed_test_df_filepath), exist_ok=True)

            # Save transformed DataFrames as CSVs
            train_df_final.to_csv(self.config.transformed_train_df_filepath, index=False)
            test_df_final.to_csv(self.config.transformed_test_df_filepath, index=False)
            logging.info(f"Train DataFrame saved at: {self.config.transformed_train_df_filepath}")
            logging.info(f"Test DataFrame saved at: {self.config.transformed_test_df_filepath}")


            # Also save numpy arrays for modeling
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            os.makedirs(os.path.dirname(self.config.transformed_preprocessor_obj_filepath), exist_ok=True)
            joblib.dump(preprocessor_obj, self.config.transformed_preprocessor_obj_filepath)

            os.makedirs(os.path.dirname(self.config.transformed_train_arr_filepath), exist_ok=True)
            np.save(self.config.transformed_train_arr_filepath, train_arr)
            os.makedirs(os.path.dirname(self.config.transformed_test_arr_filepath), exist_ok=True)
            np.save(self.config.transformed_test_arr_filepath, test_arr)

            # Return artifact object
            artifact = DataTransformationArtifact(
                transformed_train_arr_filepath=self.config.transformed_train_arr_filepath,
                transformed_test_arr_filepath=self.config.transformed_test_arr_filepath,
                transformed_preprocessor_obj_filepath=self.config.transformed_preprocessor_obj_filepath,
                transformed_train_df_filepath=self.config.transformed_train_df_filepath,
                transformed_test_df_filepath=self.config.transformed_test_df_filepath
            )

            logging.info(f"Train array saved at: {self.config.transformed_train_arr_filepath}")
            logging.info(f"Test array saved completed")
            logging.info(f"Preprocessor object completed")


            logging.info("Data transformation completed successfully.")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)

        


# ============================================================================================
# ============================================================================================
# =============== DATA_VALIDATION ============================================================
# ============================================================================================
# ============================================================================================
import os, sys
from pathlib import Path
from Churn_Pred.exception.exception import CustomException
from Churn_Pred.logger.log import logging
from Churn_Pred.utils import read_yaml_file, write_yaml_file
from Churn_Pred.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from Churn_Pred.components.data_ingestion import DataIngestionConfig, DataIngestion

import pandas as pd
from scipy.stats import ks_2samp
from dataclasses import dataclass


# ------ Data Validation Configuration ------
@dataclass
class DataValidationConfig:
    validation_status: Path = Path('artifacts/validated/validation_status.txt')
    valid_train_file_path: Path = Path('artifacts/validated/valid/valid_train.csv')
    valid_test_file_path: Path = Path('artifacts/validated/valid/valid_test.csv')
    invalid_train_file_path: Path = Path('artifacts/validated/invalid/invalid_train.csv')
    invalid_test_file_path: Path = Path('artifacts/validated/invalid/invalid_test.csv')
    drift_report_file_path: Path = Path('artifacts/validated/drift/report.yaml')
    status_file_path: Path = Path('artifacts/validated/drift/status.txt')


# ------ Data Validation class ------
class DataValidation:
    def __init__(self, 
                 config:DataValidationConfig, 
                 data_ingestion_artifact:DataIngestionArtifact
    ):
        try:
            self.config = config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file('schema.yaml')
        except Exception as e:
            raise CustomException(e,sys)

    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CustomException(e,sys)

    def validate_number_of_columns(self, dataframe:pd.DataFrame) -> bool:
        try:
            number_of_column = len(self._schema_config.get("columns", []))
            logging.info(f"Required number of columns: {number_of_column}")
            logging.info(f"DataFrame's num of columns: {len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_column:
                logging.info("✅ Column count validation passed.")
                return True
            else:
                logging.warning("❌ Column count validation failed.")
                return False

        except Exception as e:
            raise CustomException(e,sys)
        
    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        try:
            required_columns = [col["name"] for col in self._schema_config["columns"]]
            df_columns = df.columns.str.lower().tolist()  # In case you lowercase headers in ingestion
            missing_columns = list(set(required_columns) - set(df_columns))
            if missing_columns:
                logging.warning(f"Missing columns: {missing_columns}")
                return False
            logging.info("✅ All required columns are present.")
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def check_missing_values(self, df: pd.DataFrame) -> bool:
        try:
            max_missing = self._schema_config.get("general_rules", {}).get("max_missing_ratio_per_column", 0.0)
            missing_ratio = df.isnull().mean()
            over_limit = missing_ratio[missing_ratio > max_missing]
            if not over_limit.empty:
                logging.warning(f"Columns exceeding missing threshold: {over_limit.to_dict()}")
                return False
            logging.info("✅ Missing value check passed.")
            return True
        except Exception as e:
            raise CustomException(e, sys)
    
    def detect_data_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            for col in base_df.columns:
                d1 = base_df[col].dropna()
                d2 = current_df[col].dropna()

                ks_test_result = ks_2samp(d1, d2)

                if ks_test_result.pvalue >= threshold:
                    # No drift detected for this column
                    is_drift_found = False
                else:
                    # Drift detected
                    is_drift_found = True
                    status = False  # Overall drift status becomes False if any drift found

                report[col] = {
                    'pvalue': float(ks_test_result.pvalue),
                    'drift_status': is_drift_found
                }

            # Ensure directory exists before saving report
            self.config.drift_report_file_path.parent.mkdir(parents=True, exist_ok=True)
            write_yaml_file(filepath=self.config.drift_report_file_path, content=report)
            logging.info(f"Drift report saved at {self.config.drift_report_file_path}")

            return status

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            # Read Data from INGESTION ARTIFACT
            train_file_path = self.data_ingestion_artifact.train_data_path
            test_file_path = self.data_ingestion_artifact.test_data_path

            # read data
            train_df =DataValidation.read_data(train_file_path)
            test_df =DataValidation.read_data(test_file_path)

            # ---- Validation Checks ----
            error_messages = []

            # 1. Validate Number of Columns
            if not self.validate_number_of_columns(train_df):
                error_messages.append("❌ Train DataFrame does not contain all required columns.")
            if not self.validate_number_of_columns(test_df):
                error_messages.append("❌ Test DataFrame does not contain all required columns.")

            # 2. Validate Required Columns
            if not self.validate_required_columns(train_df):
                error_messages.append("❌ Train DataFrame is missing required columns.")
            if not self.validate_required_columns(test_df):
                error_messages.append("❌ Test DataFrame is missing required columns.")

            # 3. Check Missing Values
            if not self.check_missing_values(train_df):
                error_messages.append("❌ Train DataFrame contains columns with missing values exceeding threshold.")
            if not self.check_missing_values(test_df):
                error_messages.append("❌ Test DataFrame contains columns with missing values exceeding threshold.")

            # 4. Detect Data Drift
            drift_status = self.detect_data_drift(base_df=train_df, current_df=test_df)

            # Determine overall status
            validation_passed = len(error_messages) == 0 and drift_status

            # Save STATUS
            self.config.status_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.status_file_path, 'w', encoding='utf-8') as f:
                if validation_passed:
                    f.write("Validation Passed \n")
                else:
                    f.write('Validation Failed\n')
                    for msg in error_messages:
                        f.write(f"{msg}\n")
            
            # save Valid/Invalid data
            if validation_passed:
                self.config.valid_train_file_path.parent.mkdir(parents=True, exist_ok=True)
                train_df.to_csv(self.config.valid_train_file_path, index=False, header=True)
                test_df.to_csv(self.config.valid_test_file_path, index=False, header=True)
            else:
                # SAVE to INVALID 
                self.config.invalid_train_file_path.parent.mkdir(parents=True, exist_ok=True)
                train_df.to_csv(self.config.invalid_train_file_path, index=False, header=True)
                test_df.to_csv(self.config.invalid_test_file_path, index=False, header=True)

            logging.info('Data Validation Completed')

            return DataValidationArtifact(
                validation_status=validation_passed,
                valid_train_file_path=self.config.valid_train_file_path if validation_passed else None,
                valid_test_file_path=self.config.valid_test_file_path if validation_passed else None,
                invalid_test_file_path=self.config.invalid_test_file_path if not validation_passed else None,
                invalid_train_file_path=self.config.invalid_train_file_path if not validation_passed else None,
                drift_report_file_path=self.config.drift_report_file_path,
                status_file_path=self.config.status_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


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




# ============================================================================================
# ============================================================================================
# =============== MODEL_TRAINER ==============================================================
# ============================================================================================
# ============================================================================================
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
from Churn_Pred.utils import get_lift_status, get_overfit_warning


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

    # def save_model_trainer_artifact(self, artifact: ModelTrainerArtifact, filepath: str):
    #     with open(filepath, 'w') as f:
    #         json.dump(artifact.__dict__, f, indent=4)
    #     print(f"ModelTrainerArtifact saved at {filepath}")


    def train_models(self):
        logging.info(' ✅ Initiating Model Training...')
        try:
            preprocessor = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)
            train_df = pd.read_csv(self.data_transformation_artifact.transformed_train_df_filepath)
            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_df_filepath)
            logging.info('Loading preprocessor, train and test datasets...')
            
            # Split features and target
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
            
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

                # Overfitting Warning
                overfit_warning = get_overfit_warning(model_name, train_acc, test_acc, threshold=self.config.overfit_underfit_threshold)

                # Lift Status
                lift_score = self.plot_lift_gain(model_name, pipe, X_test, y_test)
                lift_status = get_lift_status(model_name, lift_score)

                # Check if test accuracy meets the expected accuracy
                if test_acc < self.config.expected_accuracy:
                    logging.warning(f"{model_name} below BAR threshold! Test accuracy = {test_acc}")

                # Collect classification metrics
                class_report = classification_report(y_test, y_test_pred, output_dict=True)
                f1 = class_report['1'].get('f1-score', 0)
                precision = class_report['1'].get('precision', 0)
                recall = class_report['1'].get('recall', 0)
                roc_auc = self.calculate_roc_auc(pipe, X_test, y_test)

                cm = confusion_matrix(y_test, y_test_pred)
                self.save_all_model_outputs(model_name, pipe, cm, roc_auc, X_test, y_test)

                cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')

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
                    "Overfitting Warning": overfit_warning,
                    "Lift Status": lift_status
                })

                trained_pipelines[model_name] = pipe

            # Create the comparison report dataframe
            report_df = pd.DataFrame(report)
            self.save_comparison_report(report_df)

            # --- Save best model ---
            best_model_row = report_df.loc[report_df['F1'].idxmax()]
            best_model_name = best_model_row['Model']
            best_pipeline = trained_pipelines[best_model_name]

            # Define model save path
            trained_model_filepath = os.path.join(self.config.trained_model_dir, f"{best_model_name}_model.pkl")

            # Ensure the directory exists
            model_dir = os.path.dirname(trained_model_filepath)
            os.makedirs(model_dir, exist_ok=True)

            # Save the best model pipeline
            joblib.dump(best_pipeline, trained_model_filepath)
            print(f"✅ Best model '{best_model_name}' saved at {trained_model_filepath}")

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
        # self.plot_lift_gain(model_name, pipeline, X_test, y_test)
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
        print(f"Top decile lift for {model_name}: {top_decile_lift}")

        path_csv = f"{model_name}_lift_gain.csv"

        # Save the lift table as CSV for further inspection
        lift_table.to_csv(path_csv, index=False)

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

            report_df.to_csv(path, index=False)
            
            logging.info(f"📊 Model comparison report saved at: {path.resolve()}")
            print("\n📊 Model Comparison Report:\n", report_df)
        except Exception as e:
            logging.error(f"❌ Failed to save model comparison report: {e}")
            raise CustomException(e,sys)
    


    # def train_models(self):
    #     logging.info(' ✅ Initiating Model Training...')
    #     try:
    #         preprocessor = joblib.load(self.data_transformation_artifact.transformed_preprocessor_obj_filepath)
    #         # Load the transformed training data
    #         # train_arr = np.load(self.data_transformation_artifact.transformed_train_arr_filepath)
    #         # test_arr = np.load(self.data_transformation_artifact.transformed_test_arr_filepath)

    #         train_df = pd.read_csv(self.data_transformation_artifact.transformed_train_df_filepath)
    #         test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_df_filepath)
    #         logging.info('Loading preprocessor, train and test datasets...')
            
    #         # Split features and target
    #         X_train = train_df.iloc[:, :-1]
    #         y_train = train_df.iloc[:, -1]
            
    #         print(f"Loading preprocessed data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
            

    #         # model_trainer.train_models(X_train, y_train, X_test, y_test)

    #         X_test = test_df.iloc[:, :-1]
    #         y_test = test_df.iloc[:, -1]

    #         print(f"Loading preprocessed data: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
            

    #         models = {
    #             'RandomForest': RandomForestClassifier(verbose=1),
    #             'DecisionTree': DecisionTreeClassifier(),
    #             'LogisticRegression': LogisticRegression(),
    #             'Xgb': XGBClassifier()
    #         }

    #         report = []
    #         trained_pipelines = {}

    #         for model_name, model in models.items():
    #             logging.info(f"🔧 Training {model_name}")

    #             pipe = Pipeline([
    #                 ('classifier', model)
    #             ])

    #             pipe.fit(X_train, y_train)

    #             # Model Predictions
    #             y_train_pred = pipe.predict(X_train)
    #             y_test_pred = pipe.predict(X_test)

    #             # Model Accuracy
    #             train_acc = accuracy_score(y_train, y_train_pred)
    #             test_acc = accuracy_score(y_test, y_test_pred)

    #             # Check Test Accuracy meets Expected Accuracy criteria or not
    #             if test_acc < self.config.expected_accuracy:
    #                 logging.warning(f"{model_name} below BAR threshold! Test accuracy = {test_acc}")

    #             # Check wheter overfit/underfit and its levels
    #             print_overfit_warning(model_name, train_acc, test_acc, threshold=self.config.overfit_underfit_threshold)
    #             # logging.info(f"SHAP values for  {model_name}: ")
    #             # self.plot_shap_summary(model_name=model_name, model=pipe, X_test=X_test)

    #             class_report = classification_report(y_test, y_test_pred, output_dict=True)
    #             f1 = class_report['1'].get('f1-score', 0)
    #             precision = class_report['1'].get('precision', 0)
    #             recall = class_report['1'].get('recall', 0)
    #             roc_auc = self.calculate_roc_auc(pipe, X_test, y_test)

    #             cm = confusion_matrix(y_test, y_test_pred)
    #             self.save_all_model_outputs(model_name, pipe, cm, roc_auc, X_test, y_test)

    #             cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')

    #             # Lift score
    #             lift_score = self.plot_lift_gain(model_name, pipe, X_test, y_test)
    #             print(f"Lift Score for the top decile: {lift_score}")

    #             report.append({
    #                 "Model": model_name,
    #                 "Train Accuracy": train_acc,
    #                 "Test Accuracy": test_acc,
    #                 "F1": f1,
    #                 "Precision": precision,
    #                 "Recall": recall,
    #                 "ROC AUC": roc_auc,
    #                 "CV Mean": cv_scores.mean(),
    #                 "CV Std": cv_scores.std()
    #             })

    #             # Keep pipeline for later use (to save best)
    #             trained_pipelines[model_name] = pipe

                

    #         report_df = pd.DataFrame(report)
    #         self.save_comparison_report(report_df)

    #         # --- Save best model and artifact ---
    #         best_model_row = report_df.loc[report_df['F1'].idxmax()]
    #         best_model_name = best_model_row['Model']
    #         best_pipeline = trained_pipelines[best_model_name]

    #         # Define model save path
    #         trained_model_filepath = os.path.join(self.config.trained_model_dir, f"{best_model_name}_model.pkl")

    #         # Ensure the directory exists
    #         model_dir = os.path.dirname(trained_model_filepath)
    #         os.makedirs(model_dir, exist_ok=True)

    #         # Save best model pipeline
    #         joblib.dump(best_pipeline, trained_model_filepath)
    #         print(f"✅ Best model '{best_model_name}' saved at {trained_model_filepath}")

            

    #         # Create ModelTrainerArtifact
    #         model_trainer_artifact = ModelTrainerArtifact(
    #             trained_model_filepath=trained_model_filepath,
    #             train_metric_filepath=self.config.train_metric_filepath,
    #             test_metric_filepath=self.config.test_metric_filepath,
    #             trained_conf_matrix=self.config.trained_conf_matrix,
    #             trained_roc_plot=self.config.trained_roc_plot,
    #             trained_shap_dir=self.config.trained_shap_dir,
    #             expected_accuracy=self.config.expected_accuracy,
    #             overfit_underfit_threshold=self.config.overfit_underfit_threshold,
    #             model_name=best_model_name,
    #             train_accuracy=best_model_row['Train Accuracy'],
    #             test_accuracy=best_model_row['Test Accuracy'],
    #             f1_score=best_model_row['F1'],
    #             precision=best_model_row['Precision'],
    #             recall=best_model_row['Recall'],
    #             roc_auc=best_model_row['ROC AUC']
    #         )

    #         # Save artifact as JSON
    #         # artifact_path = os.path.join(self.config.artifact_dir, "model_trainer_artifact.json")
    #         # self.save_model_trainer_artifact(model_trainer_artifact, artifact_path)

    #         return model_trainer_artifact
    #     except Exception as e:
    #         raise CustomException(e, sys)
    

    
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




# ============================================================================================
# ============================================================================================
# =============== MODEL_PUSHER ===============================================================
# ============================================================================================
# ============================================================================================
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
        logging.info('🚀 Starting Baseline Model Training...')
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
                    logging.info(f"🔧 Training {model_name} (Baseline)")
                    pipeline = Pipeline([
                        ('classifier', model)
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
                logging.info(f"📌 Top N feature importance saved at {important_feat_csv_path}")

            # Save final probability DataFrame
            final_prob_csv = Path("artifacts/model_trained/metrics/test_predicted_probabilities.csv")
            os.makedirs(final_prob_csv.parent, exist_ok=True)
            prob_df.to_csv(final_prob_csv, index=False)
            logging.info(f"✅ All model probabilities saved at {final_prob_csv}")

            # Save report
            report_df = pd.DataFrame(report)
            report_path = Path("artifacts/model_trained/metrics/baseline_model_comparison.csv")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_df.to_csv(report_path, index=False)

            logging.info(f"📊 Baseline comparison report saved: {report_path}")
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
            print(f"✅ Best model '{best_model_name}' saved at {trained_model_filepath}")

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
        print(f"Top decile lift for {model_name}: {top_decile_lift}")

        path_csv = self.config.lift_gain_csv_path / f"{model_name}_lift_gain.csv"

        # Save the lift table as CSV for further inspection
        lift_table.to_csv(path_csv, index=False)
        logging.info(f"✅ Saved Model Lift: {path_csv}")

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
        logging.info(f"✅ Model saved: {path}")

    def save_combined_excel(self, report_df: pd.DataFrame):
        try:
            logging.info("📒 Creating combined Excel workbook...")

            # Paths
            excel_path = Path("artifacts/model_trained/metrics/baseline_model_outputs.xlsx")
            top_feat_path = Path("artifacts/model_trained/metrics/top_features.csv")
            prob_path = Path("artifacts/model_trained/metrics/test_predicted_probabilities.csv")

            # Read all data
            top_feat_df = pd.read_csv(top_feat_path) if top_feat_path.exists() else pd.DataFrame()
            prob_df = pd.read_csv(prob_path) if prob_path.exists() else pd.DataFrame()

            # Save all sheets to one Excel
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                report_df.to_excel(writer, sheet_name="Model_Comparison", index=False)
                top_feat_df.to_excel(writer, sheet_name="Top_Features", index=False)
                prob_df.to_excel(writer, sheet_name="Test_Probabilities", index=False)

            logging.info(f"✅ Combined Excel saved: {excel_path}")

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
        plt.title(f"Top {top_n} Features - {model_name} (Baseline)")
        plt.gca().invert_yaxis()
        os.makedirs(self.config.important_feat_plot, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.important_feat_plot, f"{model_name}_feature_importance.png"))
        plt.close()
        logging.info(f"✅ Feature importance plot saved for {model_name}")
        return feature_df




        
if __name__ == "__main__":
    # Step 1: Ingest the data
    ingestion_config = DataIngestionConfig()
    ingestion = DataIngestion(ingestion_config)
    ingestion_artifact = ingestion.initiate_data_ingestion()
    logging.info(f"Data Ingestion Artifact: {ingestion_artifact}")

    # Step 2: Validate the data using the artifact from ingestion
    validation_config = DataValidationConfig()
    validation = DataValidation(config=validation_config, data_ingestion_artifact=ingestion_artifact)
    validation_artifact = validation.initiate_data_validation()
    logging.info(f"Data Validation Artifact: {validation_artifact}")

    # Step 3: Transform the data using the artifact from Data VALIDATION
    transformation_config = DataTransformationConfig()
    transformation = DataTransformation(config=transformation_config, data_valid_artifact=validation_artifact)
    transformation_artifact = transformation.initiate_data_transformation()
    logging.info(f"Data Transformation Artifact: {transformation_artifact}")

    # Step 4: Training model data using the artifact from Data TRANSFORMATION
    mlflow.set_tracking_uri("file:./mlruns")  # or use remote server
    mlflow.set_experiment("ChurnPredictionModels")
    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(config=model_trainer_config, data_transformation_artifact=transformation_artifact)
    model_trainer_artifact = model_trainer.train_baseline_models()
    logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")