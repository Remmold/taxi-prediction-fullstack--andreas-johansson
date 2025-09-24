from importlib.resources import files

ORIGINAL_CSV_PATH = files("taxipred").joinpath("data/taxi_trip_pricing.csv")
ALTERED_CSV_PATH = files("taxipred").joinpath("data/altered_taxi_trip_pricing.csv")

COLUMNS = {
    "DISTANCE": "Trip_Distance_km",
    "BASE_FARE": "Base_Fare",
    "KM_RATE": "Per_Km_Rate",
    "MIN_RATE": "Per_Minute_Rate", 
    "DURATION": "Trip_Duration_Minutes",
    "PRICE": "Trip_Price"
}

# DATA_PATH = Path(__file__).parents[1] / "data"