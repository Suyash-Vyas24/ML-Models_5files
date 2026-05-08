from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load the AI models we created
# Note: These files must be in the same folder on GitHub
range_model = joblib.load('ev_range_model.pkl')
soh_model = joblib.load('ev_soh_model.pkl')

app = FastAPI()

# Allow the internet to talk to this code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define what data is coming in
class EVTelemetry(BaseModel):
    soc_percent: float
    speed_kmh: float
    current_a: float
    batt_temp_c: float
    cycles: int
    age_months: int
    fast_charge_ratio: float

@app.post("/api/predict")
def predict_ev_status(data: EVTelemetry):
    # Setup data for the AI
    range_features = np.array([[data.soc_percent, data.speed_kmh, data.current_a, data.batt_temp_c]])
    soh_features = np.array([[data.cycles, data.age_months, data.fast_charge_ratio, data.batt_temp_c]])
    
    # Make Predictions
    pred_range = range_model.predict(range_features)[0]
    pred_soh = soh_model.predict(soh_features)[0]
    
    # Generate Alerts
    alerts = []
    if data.batt_temp_c > 45:
        alerts.append({"type": "critical", "msg": "High Battery Temp > 45°C"})
    if data.soc_percent < 15:
        alerts.append({"type": "warning", "msg": "Low Battery"})

    # Send data back to the website
    return {
        "status": "success",
        "data": {
            "predicted_range_km": round(pred_range, 1),
            "predicted_soh_percent": round(pred_soh, 1),
            "active_alerts": alerts
        }
    }