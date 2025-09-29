# app.py
import streamlit as st
import googlemaps
from datetime import datetime, time
import requests

# Import your custom helper function
from taxipred.utils.helpers import post_api_endpoint

# Use the same API key import method you provided
from dotenv import load_dotenv
import os
load_dotenv() # take environment variables from .env.
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


# --- Initialize Google Maps Client ---
try:
    gmaps = googlemaps.Client(key=API_KEY)
except Exception as e:
    st.error(f"Could not connect to Google Maps. Check your API Key. Error: {e}")
    st.stop()


# --- Helper function for clarity ---
def get_time_of_day(hour):
    """Maps an hour (0-23) to Morning, Afternoon, Evening, or Night."""
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

# --- STREAMLIT APP ---
st.title("Simple Taxi Fare Estimator")

# 1. Get User Inputs
st.header("Trip Details")
start_address = st.text_input("📍 Start Location", "Stockholm Central Station")
end_address = st.text_input("🏁 End Location", "Vasa Museum, Stockholm")

pickup_date = st.date_input("Pickup Date")
pickup_time = st.time_input("Pickup Time")
departure_time = datetime.combine(pickup_date, pickup_time)

# Inputs required by your Pydantic model are now grouped
col1, col2, col3 = st.columns(3)
with col1:
    weather = st.selectbox(
        "🌦️ Weather",
        ("Clear", "Rain", "Snow") # Using options from your previous code
    )
with col2:
    # ADDED: Passenger Count input
    passenger_count = st.number_input("👤 Passengers", min_value=1, max_value=8, value=1)
with col3:
    # ADDED: Traffic Conditions input
    traffic_conditions = st.selectbox(
        "🚦 Traffic",
        ("Low", "Medium", "High") # Options from your CSV file
    )


# 2. The "Go" Button
if st.button("Estimate Trip", type="primary"):
    if not start_address or not end_address:
        st.warning("Please enter both a start and end address.")
    else:
        try:
            day_type = "Weekday" if pickup_date.weekday() < 5 else "Weekend"
            time_of_day = get_time_of_day(pickup_time.hour)
            
            with st.spinner("Finding locations and calculating distance..."):
                start_place_result = gmaps.find_place(start_address, 'textquery')
                end_place_result = gmaps.find_place(end_address, 'textquery')

                if not start_place_result['candidates'] or not end_place_result['candidates']:
                    st.error("Could not find one or both locations. Please be more specific.")
                else:
                    start_place_id = start_place_result['candidates'][0]['place_id']
                    end_place_id = end_place_result['candidates'][0]['place_id']

                    matrix_result = gmaps.distance_matrix(
                        origins=[f"place_id:{start_place_id}"],
                        destinations=[f"place_id:{end_place_id}"],
                        mode="driving",
                        departure_time=departure_time
                    )
                
                    element = matrix_result['rows'][0]['elements'][0]
                    
                    if element['status'] == 'OK':
                        trip_distance_text = element['distance']['text']
                        trip_distance_km = element['distance']['value'] / 1000.0

                        if 'duration_in_traffic' in element:
                            trip_duration_text = element['duration_in_traffic']['text']
                            duration_note = "(with traffic estimate)"
                        else:
                            trip_duration_text = element['duration']['text']
                            duration_note = ""

                        st.success("Trip information calculated!")
                        st.info(f"🚗 **Distance:** {trip_distance_text}")
                        st.info(f"⏳ **Estimated Time:** {trip_duration_text} {duration_note}")
                        
                        # --- FINAL API CALL TO YOUR BACKEND ---
                        st.subheader("Prediction:")
                        with st.spinner("Getting fare prediction from your model..."):
                            # Create the payload with keys matching your Pydantic `Trip` class
                            payload = {
                                "Trip_Distance_km": trip_distance_km,
                                "Time_of_Day": time_of_day,
                                "Day_of_Week": day_type,
                                "Passenger_Count": passenger_count,
                                "Traffic_Conditions": traffic_conditions,
                                "Weather": weather
                            }
                            
                            # Use your helper function to make the POST request
                            response = post_api_endpoint(data=payload)
                            response.raise_for_status() # Raise an error for bad status codes
                            
                            predicted_fare = response.json()
                            st.success(f"## 💵 Predicted Fare: ${predicted_fare:.2f}")
                        
                    else:
                        st.error(f"Could not calculate a route. Reason: {element['status']}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the prediction service. Please ensure it is running. Error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")