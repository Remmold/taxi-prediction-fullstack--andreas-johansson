from pydantic import BaseModel

class Trip(BaseModel):
    Trip_Distance_km:float
    Time_of_Day:str
    Day_of_Week:str
    Passenger_Count:int
    Traffic_Conditions:str
    Weather:str