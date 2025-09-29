import streamlit as st
import googlemaps
import requests
from datetime import datetime, time

# Use the same API key import method you provided
from dotenv import load_dotenv
import os
load_dotenv() # take environment variables from .env.
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
FASTAPI_URL = "http://127.0.0.1:8000/predict" # Your FastAPI backend URL

# Initialize the Google Maps client
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

# Combine date and time inputs for the API call
pickup_date = st.date_input("Pickup Date")
pickup_time = st.time_input("Pickup Time")
departure_time = datetime.combine(pickup_date, pickup_time)

# New weather input for your ML model
weather = st.selectbox(
    "🌦️ Current Weather",
    ("Sunny", "Cloudy", "Rain", "Light Rain", "Snow")
)

# 2. The "Go" Button
if st.button("Estimate Trip", type="primary"):
    if not start_address or not end_address:
        st.warning("Please enter both a start and end address.")
    else:
        try:
            # --- NEW: Determine time_of_day and day_type ---
            # weekday() returns 0 for Monday, 6 for Sunday.
            day_type = "Weekday" if pickup_date.weekday() < 5 else "Weekend"
            time_of_day = get_time_of_day(pickup_time.hour)
            
            with st.spinner("Finding locations and calculating distance..."):
                # Step 1: Use Places API to find the specific place ID for each address
                start_place_result = gmaps.find_place(start_address, 'textquery')
                end_place_result = gmaps.find_place(end_address, 'textquery')

                if not start_place_result['candidates'] or not end_place_result['candidates']:
                    st.error("Could not find one or both locations. Please be more specific.")
                else:
                    start_place_id = start_place_result['candidates'][0]['place_id']
                    end_place_id = end_place_result['candidates'][0]['place_id']

                    # Step 2: Use Distance Matrix API with the place IDs
                    matrix_result = gmaps.distance_matrix(
                        origins=[f"place_id:{start_place_id}"],
                        destinations=[f"place_id:{end_place_id}"],
                        mode="driving",
                        departure_time=departure_time
                    )
                
                    # 3. Extract and save the results into variables
                    element = matrix_result['rows'][0]['elements'][0]
                    
                    if element['status'] == 'OK':
                        # The road distance (saved in a variable)
                        trip_distance_text = element['distance']['text']
                        # CONVERT meters to kilometers
                        trip_distance_km = element['distance']['value'] / 1000.0

                        # The estimated trip duration (saved in a variable)
                        if 'duration_in_traffic' in element:
                            trip_duration_text = element['duration_in_traffic']['text']
                            # CONVERT seconds to minutes
                            trip_duration_minutes = element['duration_in_traffic']['value'] / 60.0
                            duration_note = "(with traffic estimate)"
                        else:
                            trip_duration_text = element['duration']['text']
                            # CONVERT seconds to minutes
                            trip_duration_minutes = element['duration']['value'] / 60.0
                            duration_note = ""

                        # Display the results
                        st.success("Trip estimate successful!")
                        st.info(f"🚗 **Distance:** {trip_distance_text}")
                        st.info(f"⏳ **Estimated Time:** {trip_duration_text} {duration_note}")
                        
                        # You now have the variables ready for your ML model
                        st.subheader("Data for ML Model:")
                        st.write(f"- Distance in km: `{trip_distance_km:.2f}`")
                        st.write(f"- Duration in minutes: `{trip_duration_minutes:.2f}`")
                        st.write(f"- Weather condition: `{weather}`")
                        st.write(f"- Day Type: `{day_type}`")
                        st.write(f"- Time of Day: `{time_of_day}`")
                        
                        # Here, you would typically send these variables to your FastAPI backend
                        # payload = {
                        #     "trip_distance_km": trip_distance_km,
                        #     "trip_duration_minutes": trip_duration_minutes,
                        #     "weather": weather,
                        #     "day_type": day_type,
                        #     "time_of_day": time_of_day,
                        # }
                        # response = requests.post(FASTAPI_URL, json=payload)
                        # st.write(f"Predicted Fare: ${response.json()['predicted_fare']:.2f}")
                    else:
                        st.error(f"Could not calculate a route. Reason: {element['status']}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

