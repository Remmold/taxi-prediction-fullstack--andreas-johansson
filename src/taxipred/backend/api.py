from fastapi import FastAPI
from taxipred.backend.data_processing import TaxiData,Trip
from taxipred.utils.constants import CLEANED_CSV_PATH

app = FastAPI()

taxi_data = TaxiData(CLEANED_CSV_PATH)

@app.get("/taxi/")
async def read_taxi_data():
    return taxi_data.to_json()

@app.post("/taxi/predict")
async def predict(trip:Trip) ->int :
    prediction = taxi_data.predict_trip_price(trip)
    return prediction
