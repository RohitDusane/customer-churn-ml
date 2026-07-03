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
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self):
        logging.info('🚀 Initiating Data Transformation Step...\n')
        try:
            # Load validated train and test data
            logging.info("Loading Validated datasets...")
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
            # logging.info(f"Train DataFrame saved at: {self.config.transformed_train_df_filepath}")
            # logging.info(f"Test DataFrame saved at: {self.config.transformed_test_df_filepath}")
            logging.info(f"Train DataFrame saved completed")
            logging.info(f"Test DataFrame saved completed")

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

            logging.info(f"Train array saved completed")
            logging.info(f"Test array saved completed")
            logging.info(f"Preprocessor object saved completed")

            logging.info("Data Transformation Completed Successfully.")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)