from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# import numpy as np
import pandas as pd
import joblib
import logging
import json
import datetime
import os
import uvicorn

# ---------- Setup Logging ----------
log_dir = "prediction_app_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ---------- Load Preprocessor and Model ----------
PREPROCESSOR_PATH = "artifacts/transformed/preprocessor.pkl"
MODEL_PATH = "artifacts/model_trained/best_models/RandomForest_model.pkl"

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    logging.error(f"❌ Failed to load model/preprocessor: {e}")
    raise e

# ---------- Initialize FastAPI ----------
app = FastAPI(title="Churn Prediction API")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- Input Schema ----------
class CustomerData(BaseModel):
    CreditScore: float
    Geography: int  # Encoded: France=0, Germany=1, Spain=2
    Gender: int     # Encoded: Female=0, Male=1
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "message": message})



@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/feedback")
async def submit_feedback(feedback: str = Form(...)):
    with open("feedback_log.txt", "a") as f:
        f.write(feedback + "\n")
    return RedirectResponse(url="/?message=Thank you for your feedback!", status_code=302)

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    CreditScore: float = Form(...),
    Geography: int = Form(...),
    Gender: int = Form(...),
    Age: float = Form(...),
    Tenure: int = Form(...),
    Balance: float = Form(...),
    NumOfProducts: int = Form(...),
    HasCrCard: int = Form(...),
    IsActiveMember: int = Form(...),
    EstimatedSalary: float = Form(...)
):
    try:
        # ✅ Build the input_dict dynamically from form inputs
        input_dict = {
            "creditscore": CreditScore,
            "geography": Geography,
            "gender": Gender,
            "age": Age,
            "tenure": Tenure,
            "balance": Balance,
            "numofproducts": NumOfProducts,
            "hascrcard": HasCrCard,
            "isactivemember": IsActiveMember,
            "estimatedsalary": EstimatedSalary
        }

        # ✅ Create DataFrame and ensure column order matches training
        input_df = pd.DataFrame([input_dict])

        # 🔄 Optional: enforce column order if needed
        expected_order = [
            "creditscore", "geography", "gender", "age", "tenure",
            "balance", "numofproducts", "hascrcard", "isactivemember", "estimatedsalary"
        ]
        input_df = input_df[expected_order]

        # ✅ Transform and predict
        transformed = preprocessor.transform(input_df)

        # ✅ Use custom threshold for prediction
        probability = model.predict_proba(transformed)[0][1]
        threshold = 0.3
        prediction = 1 if probability >= threshold else 0

        print("🔎 Probability of churn:", probability)
        print("📉 Model prediction:", prediction)


        # ✅ Log prediction
        log_data = {
            "input": input_dict,
            "prediction": int(prediction),
            "churn_probability": round(probability, 4),
            "timestamp": datetime.datetime.now().isoformat()
        }
        logging.info("Prediction Log: %s", json.dumps(log_data))

        # ✅ Send result to frontend
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "prediction": int(prediction),
                "churn_probability": round(probability, 4)
            }
        })

    except Exception as e:
        logging.error(f"Form prediction failed: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"error": str(e)}
        })


# ---------- Main Entrypoint ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)