import joblib
import pandas as pd 
from pydantic import BaseModel
from fastapi import FastAPI

# Load the modal
with open("adult_model.pkl", 'rb') as f: 
    model = joblib.load(f)

# FastAPI App
app = FastAPI(title="Income Prediction API")

# Create Schema
class AdultInput(BaseModel):
    age : int 
    workclass : str
    fnlwgt : int 
    education : str
    education_num : int
    martial_status : str 
    occupation : str
    relationship : str
    race : str 
    sex : str
    capital_gain : int
    capital_loss : int 
    hours_per_week : int
    native_country : str

# Home Endopint

@app.get("/")
def home():
    return{"Message": "Adult Income prediction API running"}

@app.post("/predict")
def predict(data: AdultInput):

    input_data = pd.DataFrame([data.dict()])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": str(prediction)
    }