from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.routes import router  # ✅ Import your modular router
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipelin

app = FastAPI(title="Churn Prediction App")

# Static files (CSS)
app.mount("/static", StaticFiles(directory="app/static/css"), name="static")

# Include routes
app.include_router(router)







# from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Query
# from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import pandas as pd
# import joblib
# import os
# import json
# import logging
# import datetime
# from uuid import uuid4

# # Set up logging
# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(filename="logs/predictions.log", level=logging.INFO, format="%(asctime)s - %(message)s")
# # Create logs/feedback directory if not exists
# os.makedirs("logs/feedback", exist_ok=True)

# # Initialize FastAPI
# app = FastAPI(title="Churn Prediction App")
# app.mount("/static", StaticFiles(directory="app/static/css"), name="static")
# templates = Jinja2Templates(directory="app/templates")

# # Load model and preprocessor
# preprocessor = joblib.load("artifacts/transformed/preprocessor.pkl")
# model = joblib.load("artifacts/model_trained/best_models/XGBoost_tuned_model.pkl")

# # Root page redirects to /predict
# @app.get("/", response_class=HTMLResponse)
# async def root():
#     return RedirectResponse(url="/predict")

# # Health check
# @app.get("/health")
# def health():
#     return {"status": "healthy"}

# # Prediction Page
# @app.get("/predict", response_class=HTMLResponse)
# async def predict_page(request: Request):
#     result = None
#     return templates.TemplateResponse("predict.html", {"request": request, "result": result})


# @app.post("/predict_form", response_class=HTMLResponse)
# async def predict_single_form(
#     request: Request,
#     CreditScore: float = Form(...),
#     Geography: int = Form(...),
#     Gender: int = Form(...),
#     Age: float = Form(...),
#     Tenure: int = Form(...),
#     Balance: float = Form(...),
#     NumOfProducts: int = Form(...),
#     HasCrCard: int = Form(...),
#     IsActiveMember: int = Form(...),
#     EstimatedSalary: float = Form(...)
# ):
#     try:
#         input_data = {
#             "creditscore": CreditScore,
#             "geography": Geography,
#             "gender": Gender,
#             "age": Age,
#             "tenure": Tenure,
#             "balance": Balance,
#             "numofproducts": NumOfProducts,
#             "hascrcard": HasCrCard,
#             "isactivemember": IsActiveMember,
#             "estimatedsalary": EstimatedSalary,
#         }

#         df = pd.DataFrame([input_data])
#         df = df[[
#             "creditscore", "geography", "gender", "age", "tenure",
#             "balance", "numofproducts", "hascrcard", "isactivemember", "estimatedsalary"
#         ]]

#         transformed = preprocessor.transform(df)
#         prob = model.predict_proba(transformed)[0][1]
#         prediction = int(prob >= 0.3)

#         result = {
#             "prediction": prediction,
#             "churn_probability": round(prob * 100, 2),  # percentage
#         }

#         logging.info(json.dumps({
#             "input": input_data,
#             "prediction": prediction,
#             "churn_probability": round(prob, 4),
#             "timestamp": datetime.datetime.now().isoformat()
#         }))

#         return templates.TemplateResponse("predict.html", {
#             "request": request,
#             "result": result,
#             "form_data": input_data
#         })

#     except Exception as e:
#         return templates.TemplateResponse("predict.html", {
#             "request": request,
#             "result": {"error": str(e)},
#             "form_data": {}  # optionally send back empty or partial form data
#         })



# # Batch prediction route
# @app.post("/predict_batch", response_class=JSONResponse)
# async def predict_batch(file: UploadFile = File(...)):
#     try:
#         df = pd.read_csv(file.file)

#         required_cols = [
#             "creditscore", "geography", "gender", "age", "tenure",
#             "balance", "numofproducts", "hascrcard", "isactivemember", "estimatedsalary"
#         ]

#         if not all(col in df.columns for col in required_cols):
#             return JSONResponse({"error": f"Missing columns. Expected: {required_cols}"}, status_code=400)

#         df = df[required_cols]
#         transformed = preprocessor.transform(df)
#         probs = model.predict_proba(transformed)[:, 1]
#         preds = (probs >= 0.3).astype(int)

#         df["churn_probability"] = [round(float(p), 4) for p in probs]
#         df["prediction"] = preds

#         # Save result
#         os.makedirs("predictions", exist_ok=True)
#         file_id = uuid4().hex
#         file_path = f"predictions/{file_id}.csv"
#         df.to_csv(file_path, index=False)

#         return {
#             "download_url": f"/download_csv/{file_id}.csv",
#             "message": "✅ Batch prediction complete. Click below to download results.",
#             "num_predictions": len(df)
#         }

#     except Exception as e:
#         logging.error(f"Batch prediction failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Download CSV
# @app.get("/download_csv/{filename}", response_class=FileResponse)
# async def download_csv(filename: str):
#     file_path = os.path.join("predictions", filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path, media_type="text/csv", filename=filename)
#     raise HTTPException(status_code=404, detail="File not found")


# # Feedback route
# @app.post("/feedback")
# async def feedback(feedback: str = Form(...)):
#     log_path = "logs/feedback/feedback.txt"
#     with open(log_path, "a") as f:
#         f.write(f"{datetime.datetime.now().isoformat()}: {feedback}\n")
#     return RedirectResponse(url="/predict", status_code=303)
