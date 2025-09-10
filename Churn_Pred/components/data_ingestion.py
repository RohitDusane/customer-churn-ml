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
        logging.info('🚀 Starting Data Ingestion step...\n')

        try:
            # Step 1: Ensure the raw data exists
            if not self.config.raw_data_path.exists():
                logging.error(f"Raw data file not found at {self.config.raw_data_path}")
                raise FileNotFoundError(f"Missing raw data file at {self.config.raw_data_path}")

            # Step 2: Read the raw data
            logging.info("  Reading raw data...")
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
            return artifact
            logging.info('Data ingestion step completed successfully. \n')

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)











# import os, sys
# from pathlib import Path
# from Churn_Pred.exception.exception import CustomException
# from Churn_Pred.logger.log import logging

# from Churn_Pred.utils import *
# from Churn_Pred.entity.artifact_entity import DataIngestionArtifact

# import pandas as pd

# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass

# # ----- Data Ingestion Configuration -------
# @dataclass
# class DataIngestionConfig:
#     # train_data_path:Path = os.path.join('artifacts', 'data', "train.csv")
#     # test_data_path:Path = os.path.join('artifacts','data', 'test.csv')
#     # raw_data_path:Path = os.path.join('artifacts','rawdata.csv')
#     raw_data_path: Path = Path('artifacts/rawdata.csv')
#     train_data_path: Path = Path('artifacts/ingested/train.csv')
#     test_data_path: Path = Path('artifacts/ingested/test.csv')
#     test_size: float = 0.2
#     random_state: int = 24
#     num_train_rows: int = 0
#     num_test_rows: int = 0

# # ----- Data Ingestion Class -------

# class DataIngestion:
#     def __init__(self, config: DataIngestionConfig):
#         self.config = config

#     def initiate_data_ingestion(self) -> DataIngestionArtifact:
#         logging.info('Starting DI step')
#         try:
#             # Read data
#             logging.info("Reading raw data...")
#             df = pd.read_csv(self.config.raw_data_path)

#             # Drop unwanted columns early
#             drop_columns = ['rownumber', 'customerid', 'surname']
#             df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
#             logging.info(f"Dropped columns: {drop_columns}")

#             # Lowercase all column names
#             df = lowercase_columns(df)
#             logging.info(f"Dataset shape: {df.shape}")
#             # logging.info(f"Sample columns: {df.columns.tolist()}")
            
#             # Save raw data            
#             self.config.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
#             df.to_csv(self.config.raw_data_path, index=False, encoding='utf-8')
#             logging.info(f"Raw data saved at {self.config.raw_data_path}")

            
#             # # Save raw data
#             # self.config.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
#             # train_set.to_csv(self.config.train_data_path, index=False)
#             # os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
#             # df.to_csv(self.config.raw_data_path, index=False, encoding='utf-8')
#             # logging.info(f"Raw data saved at {self.config.raw_data_path}")
#             # logging.info(f"Raw data saved at {str(self.config.raw_data_path)}")
            
#             # Split the data into `train` and `test`
#             train_set, test_set = train_test_split(df, 
#                                                    test_size = self.config.test_size,
#                                                    random_state=self.config.random_state)
#             # os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
#             # os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)
#             # train_set.to_csv(self.config.train_data_path,index=False, header=True)
#             # test_set.to_csv(self.config.test_data_path,index=False, header=True)
#             # Save train/test data
#             self.config.train_data_path.parent.mkdir(parents=True, exist_ok=True)
#             self.config.test_data_path.parent.mkdir(parents=True, exist_ok=True)

#             train_set.to_csv(self.config.train_data_path, index=False)
#             test_set.to_csv(self.config.test_data_path, index=False)

#             logging.info(f"Train data saved at {self.config.train_data_path}")
#             logging.info(f"Test data saved at {self.config.test_data_path}")

#             logging.info('Train-test split completed and data saved.')

#             artifact = DataIngestionArtifact(
#                 train_data_path=self.config.train_data_path,
#                 test_data_path=self.config.test_data_path,
#                 num_train_rows=train_set.shape[0],
#                 num_test_rows=test_set.shape[0]
#             )

#             return artifact

#         except Exception as e:
#             raise CustomException(e,sys)
        
# # if __name__ == "__main__":

# #     data_ingestion_config = DataIngestionConfig()
# #     data_ingestion = DataIngestion(config=data_ingestion_config)
# #     data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
#     # data_ingestion = DataIngestion(config=data_ingestion_config)
#     # ingestion_artifact = data_ingestion.initiate_data_ingestion()
#     # obj = DataIngestion()
#     # train_path, test_path = obj.initiate_data_ingestion()
#     # print(f"Train data at: {train_path}")
#     # print(f"Test data at: {test_path}")
# # if __name__ == "__main__":
# #     config = DataIngestionConfig()
# #     ingestion = DataIngestion(config)
# #     artifact = ingestion.initiate_data_ingestion()
# #     logging.info(f"Data Ingestion Artifact: {artifact}")