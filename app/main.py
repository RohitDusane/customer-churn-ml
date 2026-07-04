from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router  # ✅ Import your modular router

app = FastAPI(
    title="Customer Churn Prediction",
    description="ML-powered real-time and batch customer churn prediction system.",
    version="2.0.0",
)

# Static files (CSS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routes
app.include_router(router)