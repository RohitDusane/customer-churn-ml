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
        logging.info('🚀 Starting Data Validation step...\n')
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
                logging.info('Data Validation Passed')
            else:
                # SAVE to INVALID 
                self.config.invalid_train_file_path.parent.mkdir(parents=True, exist_ok=True)
                train_df.to_csv(self.config.invalid_train_file_path, index=False, header=True)
                test_df.to_csv(self.config.invalid_test_file_path, index=False, header=True)
                logging.info('Data Validation Failed')

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


