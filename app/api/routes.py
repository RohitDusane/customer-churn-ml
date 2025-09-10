from fastapi import APIRouter, Request, Form, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from datetime import datetime
from app.models.customer import CustomerData
from app.services.predictor import predict_single, predict_batch

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str = None):
    return templates.TemplateResponse("home.html", {"request": request, "message": message})

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.post("/feedback")
async def submit_feedback(feedback: str = Form(...)):
    with open("app/logs/feedback_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {feedback}\n")
    return RedirectResponse(url="/?message=Thank you for your feedback!", status_code=302)

@router.post("/predict_form", response_class=HTMLResponse)
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
        customer = CustomerData(
            CreditScore=CreditScore, Geography=Geography, Gender=Gender,
            Age=Age, Tenure=Tenure, Balance=Balance, NumOfProducts=NumOfProducts,
            HasCrCard=HasCrCard, IsActiveMember=IsActiveMember, EstimatedSalary=EstimatedSalary
        )
        prediction, probability = predict_single(customer)
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "result": {
                "prediction": prediction,
                "churn_probability": probability
            }
        })
    except Exception as e:
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "result": {"error": str(e)}
        })

@router.post("/predict_batch")
async def batch_prediction(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        filename, count = predict_batch(df)
        return JSONResponse({
            "message": "Batch prediction successful",
            "download_url": f"/download_batch_csv?filename={filename}",
            "num_predictions": count
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_batch_csv")
async def download_csv(filename: str = Query(...)):
    path = f"app/batch_files/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=filename)



@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# @router.get("/predict", response_class=HTMLResponse)
# async def predict_get(request: Request):
#     result = None
#     return templates.TemplateResponse("predict.html", {"request": request, "result": result})

@router.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request, "result": None})
