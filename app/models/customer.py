from pydantic import BaseModel

class CustomerData(BaseModel):
    CreditScore: float
    Geography: int  # France=0, Germany=1, Spain=2
    Gender: int     # Female=0, Male=1
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
