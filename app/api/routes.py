"""
app/api/routes.py — All HTTP routes.

Fixes applied:
  1. predict_form now returns form_data on both success and error
     so the form fields repopulate instead of resetting.
  2. Added POST /api/predict JSON endpoint for fetch()-based calls
     from the JS frontend (no page reload required).
  3. /health returns model load status so monitoring can detect failures.
"""
from fastapi import APIRouter, Request, Form, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from datetime import datetime
from app.models.customer import CustomerData
from app.services.predictor import predict_single, predict_batch
from app.core.config import model, preprocessor

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


# ── Pages ────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str = None):
    return templates.TemplateResponse("home.html", {
        "request": request,
        "message": message
    })


@router.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "result": None,
        "form_data": {}
    })


@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat(),
    }


# ── Single prediction (form POST — server-rendered, full page) ─────────────

@router.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    CreditScore: float = Form(...),
    Geography: int   = Form(...),
    Gender: int      = Form(...),
    Age: float       = Form(...),
    Tenure: int      = Form(...),
    Balance: float   = Form(...),
    NumOfProducts: int  = Form(...),
    HasCrCard: int      = Form(...),
    IsActiveMember: int = Form(...),
    EstimatedSalary: float = Form(...),
):
    # Reconstruct form_data so the template can repopulate fields on error
    form_data = dict(
        CreditScore=CreditScore, Geography=Geography, Gender=Gender,
        Age=Age, Tenure=Tenure, Balance=Balance,
        NumOfProducts=NumOfProducts, HasCrCard=HasCrCard,
        IsActiveMember=IsActiveMember, EstimatedSalary=EstimatedSalary,
    )
    try:
        customer = CustomerData(**form_data)
        prediction, probability = predict_single(customer)

        # Derive risk tier for richer template display
        if probability >= 70:
            risk = "High"
        elif probability >= 40:
            risk = "Medium"
        else:
            risk = "Low"

        result = {
            "prediction": prediction,
            "churn_probability": probability,
            "risk_tier": risk,
        }
    except Exception as exc:
        result = {"error": str(exc)}

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "result": result,
        "form_data": form_data,
    })


# ── Single prediction (JSON API — called by JS fetch, no page reload) ────────

@router.post("/api/predict")
async def api_predict(customer: CustomerData):
    """
    JSON endpoint for the JS frontend.
    The new predict.html uses fetch() to call this so predictions
    appear without a full page reload.
    """
    try:
        prediction, probability = predict_single(customer)
        risk = "High" if probability >= 70 else "Medium" if probability >= 40 else "Low"
        return {
            "prediction": prediction,
            "churn_probability": probability,
            "risk_tier": risk,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Batch prediction ──────────────────────────────────────────────────────────

@router.post("/predict_batch")
async def batch_prediction(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        filename, count = predict_batch(df)
        return JSONResponse({
            "message": "Batch prediction successful",
            "download_url": f"/download_batch_csv?filename={filename}",
            "num_predictions": count,
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/download_batch_csv")
async def download_csv(filename: str = Query(...)):
    path = f"app/batch_files/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=filename)


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(feedback: str = Form(...)):
    os.makedirs("app/logs", exist_ok=True)
    with open("app/logs/feedback_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {feedback}\n")
    return RedirectResponse(url="/?message=Thank you for your feedback!", status_code=302)