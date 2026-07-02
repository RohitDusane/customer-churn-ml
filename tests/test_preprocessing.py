import pandas as pd
import pytest
from Churn_Pred.components.data_ingestion import (DataIngestion, DataIngestionConfig,)

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
