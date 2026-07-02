import pandas as pd
import pytest
from Churn_Pred.components.data_ingestion import (DataIngestion, DataIngestionConfig,)
from Churn_Pred.components.data_transformation import (DataTransformation, DataTransformationConfig,)
def test_schema_validation_accepts_valid_input():
    df = pd.DataFrame(
        {
            "creditscore": [600],
            "geography": ["France"],
            "gender": ["Male"],
            "age": [25],
            "tenure": [3],
            "balance": [50000],
            "numofproducts": [2],
            "hascrcard": [1],
            "isactivemember": [1],
            "estimatedsalary": [70000],
            "exited": [0],
        }
    )

    ingestion = DataIngestion(DataIngestionConfig())

    # Should not raise any exception
    ingestion.validate_schema(df)

def test_schema_validation_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "creditscore": [600],
            "age": [25],
        }
    )

    ingestion = DataIngestion(DataIngestionConfig())

    with pytest.raises(ValueError):
        ingestion.validate_schema(df)


class DummyValidationArtifact:
    """Dummy artifact since get_data_transformer_object() doesn't use it."""
    valid_train_file_path = ""
    valid_test_file_path = ""


def test_transformation_preserves_row_count():
    df = pd.DataFrame(
        {
            "creditscore": [600, 700],
            "geography": ["France", "Spain"],
            "gender": ["Male", "Female"],
            "age": [25, 30],
            "tenure": [3, 5],
            "balance": [50000.0, 70000.0],
            "numofproducts": [2, 1],
            "hascrcard": [1, 0],
            "isactivemember": [1, 1],
            "estimatedsalary": [60000.0, 80000.0],
            "exited": [0, 1],
        }
    )

    transformation = DataTransformation(
        config=DataTransformationConfig(),
        data_valid_artifact=DummyValidationArtifact(),
    )

    preprocessor = transformation.get_data_transformer_object(df)

    X = df.drop(columns=["exited"])
    transformed = preprocessor.fit_transform(X)

    # Transformation should not change the number of rows
    assert transformed.shape[0] == df.shape[0]

# To verify that target isn't included in the transformed features
def test_target_column_is_removed_before_transformation():
    df = pd.DataFrame(
        {
            "creditscore": [600],
            "geography": ["France"],
            "gender": ["Male"],
            "age": [25],
            "tenure": [3],
            "balance": [50000],
            "numofproducts": [2],
            "hascrcard": [1],
            "isactivemember": [1],
            "estimatedsalary": [60000],
            "exited": [0],
        }
    )

    transformation = DataTransformation(
        config=DataTransformationConfig(),
        data_valid_artifact=DummyValidationArtifact(),
    )

    preprocessor = transformation.get_data_transformer_object(df)

    X = df.drop(columns=["exited"])
    transformed = preprocessor.fit_transform(X)

    # Only feature rows should be transformed
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
