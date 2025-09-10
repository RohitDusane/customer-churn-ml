import pandas as pd
import datetime
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from app.core.config import model, preprocessor, THRESHOLD, LOG_DIR, BATCH_OUTPUT_DIR
from app.models.customer import CustomerData

# Set up logging with rotation
log_file = os.path.join(LOG_DIR, "predictions.log")
handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

EXPECTED_COLUMNS = [
    "creditscore", "geography", "gender", "age", "tenure",
    "balance", "numofproducts", "hascrcard", "isactivemember", "estimatedsalary"
]

def convert_numpy(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

def predict_single(data: CustomerData) -> tuple[int, float]:
    input_dict = {k.lower(): v for k, v in data.dict().items()}
    df = pd.DataFrame([input_dict])[EXPECTED_COLUMNS]

    if model is None or preprocessor is None:
        raise RuntimeError("Model or preprocessor is not loaded.")

    transformed = preprocessor.transform(df)
    prob = float(model.predict_proba(transformed)[0][1])
    prediction = int(prob >= THRESHOLD)

    log_data = {
        "input": input_dict,
        "prediction": prediction,
        "churn_probability": round(prob, 4),
        "timestamp": datetime.datetime.now().isoformat()
    }
    logging.info("Prediction: %s", json.dumps(log_data, default=convert_numpy))

    return prediction, round(prob * 100, 2)

def predict_batch(df: pd.DataFrame) -> tuple[str, int]:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[EXPECTED_COLUMNS]

    if model is None or preprocessor is None:
        raise RuntimeError("Model or preprocessor is not loaded.")

    transformed = preprocessor.transform(df)
    probs = model.predict_proba(transformed)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    df["churn_probability"] = [round(float(p), 4) for p in probs]
    df["prediction"] = preds

    # Optional: reorder output columns
    df = df[EXPECTED_COLUMNS + ["churn_probability", "prediction"]]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_predictions_{timestamp}.csv"
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(BATCH_OUTPUT_DIR, filename), index=False)

    return filename, len(df)
