
# ------ DATA INGESTION ARTIFACT ------
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass
class DataIngestionArtifact:
    train_data_path: Path
    test_data_path: Path
    num_train_rows: int
    num_test_rows: int

# ------ DATA VALDIATION ARTIFACT ------
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: Path = None
    valid_test_file_path: Path = None
    invalid_train_file_path: Path = None
    invalid_test_file_path: Path = None
    drift_report_file_path: Path = None
    status_file_path: Path = None

# ------ DATA TRANSFORMATION ARTIFACT ------
@dataclass
class DataTransformationArtifact:
    transformed_train_arr_filepath: Path = None
    transformed_test_arr_filepath: Path = None
    transformed_preprocessor_obj_filepath : Path = None
    transformed_train_df_filepath: Path = None
    transformed_test_df_filepath: Path = None

# ------ Model Trainer ARTIFACT ------
@dataclass
class ModelTrainerArtifact:
    trained_model_filepath: Path = None
    train_metric_filepath: Path = None
    test_metric_filepath : Path = None
    trained_conf_matrix: Path = None
    trained_roc_plot: Path = None
    trained_shap_dir: Path = None
    expected_accuracy:float = 0.6
    overfit_underfit_threshold:float = 0.05

    # Add these fields
    model_name: str = None
    train_accuracy: float = None
    test_accuracy: float = None
    f1_score: float = None
    precision: float = None
    recall: float = None
    roc_auc: float = None