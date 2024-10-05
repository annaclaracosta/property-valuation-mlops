
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

# Initialize FastAPI
app = FastAPI()

# Define a property model for incoming requests
class PropertyData(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float

# Load the preprocessor and model
@app.on_event("startup")
def load_model():
    global model, preprocessor
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

# Prediction endpoint
@app.post("/predict/")
def predict(data: PropertyData):
    try:
        # Convert incoming data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Preprocess the data
        processed_data = preprocessor.transform(input_data)

        # Predict using the model
        prediction = model.predict(processed_data)

        # Return the prediction
        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
