import logging

# Assuming these classes exist in your project (DataIngestion, DataValidation, etc.)
from Churn_Pred.components.data_ingestion import DataIngestion, DataIngestionConfig
from Churn_Pred.components.data_validation import DataValidation, DataValidationConfig
from Churn_Pred.components.data_transformation import DataTransformation, DataTransformationConfig
from Churn_Pred.components.model_trainer import ModelTrainer, ModelTrainerConfig
from Churn_Pred.logger.log import logging

def run_pipeline():
    try:
        # Step 1: Ingest the data
        ingestion_config = DataIngestionConfig()
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion Artifact: {ingestion_artifact}")

        # Step 2: Validate the data
        logging.info("Starting Data Validation Step")
        validation_config = DataValidationConfig()
        validation = DataValidation(config=validation_config, data_ingestion_artifact=ingestion_artifact)
        validation_artifact = validation.initiate_data_validation()
        logging.info(f"Data Validation Artifact: {validation_artifact}")

        # Step 3: Transform the data
        logging.info("Starting Data Transformation Step")
        transformation_config = DataTransformationConfig()
        transformation = DataTransformation(config=transformation_config, data_valid_artifact=validation_artifact)
        transformation_artifact = transformation.initiate_data_transformation()
        logging.info(f"Data Transformation Artifact: {transformation_artifact}")

        # Step 4: Train the model
        logging.info("Starting Model Training Step")
        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer(config=model_trainer_config, data_transformation_artifact=transformation_artifact)
        model_trainer_artifact = model_trainer.train_models()
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

        return {
            "ingestion": ingestion_artifact,
            "validation": validation_artifact,
            "transformation": transformation_artifact,
            "model_training": model_trainer_artifact,
            "status": "success"
        }

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        return {"status": "failure", "error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # You can also adjust logging level here
    logging.info("🚀 Starting the training pipeline...")
    result = run_pipeline()
    logging.info(f"Pipeline result: {result}")