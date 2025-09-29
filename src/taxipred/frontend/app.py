import streamlit as st
import googlemaps
from datetime import datetime
import requests  # Ensure requests is imported for error handling
from dotenv import load_dotenv
import os

# Import your custom helper function from the helpers.py file
from taxipred.utils.helpers import post_api_endpoint

# --- Environment and API Key Setup ---
# Load environment variables from a .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# --- Initialize Google Maps Client ---
# Use a cached resource to prevent re-initializing the client on every rerun
@st.cache_resource
def get_gmaps_client():
    """Initializes and returns a Google Maps client instance."""
    if not API_KEY:
        st.error("GOOGLE_MAPS_API_KEY environment variable not found. Please set it in your .env file.")
        return None
    try:
        return googlemaps.Client(key=API_KEY)
    except Exception as e:
        st.error(f"Could not connect to Google Maps. Check your API Key. Error: {e}")
        return None

gmaps = get_gmaps_client()
if gmaps is None:
    st.stop()


# --- Helper Functions ---
def get_time_of_day(hour):
    """Categorizes the hour of the day."""
    if 6 <= hour < 12: return "Morning"
    elif 12 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 22: return "Evening"
    else: return "Night"

@st.cache_data(show_spinner=False)
def get_place_suggestions(_gmaps_client, input_text):
    """Cached function to fetch place autocomplete suggestions from Google Maps."""
    if not input_text or len(input_text) < 3:
        return []
    try:
        return _gmaps_client.places_autocomplete(
            input_text,
            components={'country': 'SE'}  # Restrict to Sweden for relevance
        )
    except Exception:
        return []

# --- Autocomplete Address Input Widget ---
def address_input_with_autocomplete(label, key_prefix):
    """
    Creates a text input for addresses with a dropdown for autocomplete suggestions.
    This function manages its own state using st.session_state.
    """
    input_key, suggestions_key, place_id_key = f"{key_prefix}_input", f"{key_prefix}_suggestions", f"{key_prefix}_place_id"

    if input_key not in st.session_state: st.session_state[input_key] = ""
    if suggestions_key not in st.session_state: st.session_state[suggestions_key] = []
    if place_id_key not in st.session_state: st.session_state[place_id_key] = None

    user_input = st.text_input(
        label, key=f"{key_prefix}_text_input_key",
        value=st.session_state[input_key],
        on_change=lambda: st.session_state.update({place_id_key: None}),
        placeholder="e.g., Götaplatsen, Göteborg"
    )
    st.session_state[input_key] = user_input

    if user_input and st.session_state[place_id_key] is None:
        suggestions = get_place_suggestions(gmaps, user_input)
        st.session_state[suggestions_key] = suggestions
        if suggestions:
            options = ["--Please select an address--"] + [s['description'] for s in suggestions]
            selected_address = st.selectbox(
                f"Suggestions for {label}", options=options,
                key=f"{key_prefix}_selectbox", label_visibility="collapsed"
            )
            if selected_address != "--Please select an address--":
                chosen_suggestion = next((s for s in suggestions if s['description'] == selected_address), None)
                if chosen_suggestion:
                    st.session_state[input_key] = chosen_suggestion['description']
                    st.session_state[place_id_key] = chosen_suggestion['place_id']
                    st.session_state[suggestions_key] = []
                    st.rerun()

# --- STREAMLIT APP UI ---
st.set_page_config(layout="centered")
st.title("Taxi Fare Estimator")
st.markdown("Enter your trip details below to get a fare estimate.")
st.header("1. Enter Trip Details")

col1, col2 = st.columns(2)
with col1: address_input_with_autocomplete("📍 Start Location", "start")
with col2: address_input_with_autocomplete("🏁 End Location", "end")

col3, col4, col5 = st.columns(3)
with col3:
    pickup_date = st.date_input("Pickup Date")
    weather = st.selectbox("Weather", ("Clear", "Rain", "Snow"))
with col4:
    pickup_time = st.time_input("Pickup Time")
    traffic_conditions = st.selectbox("Traffic", ("Low", "Medium", "High"))
with col5:
    passenger_count = st.number_input("Passengers", min_value=1, max_value=8, value=1)

st.divider()

# --- "Estimate" Button and LIVE Prediction Logic ---
if st.button("Estimate Trip", type="primary", use_container_width=True):
    start_place_id = st.session_state.get('start_place_id')
    end_place_id = st.session_state.get('end_place_id')

    if not start_place_id or not end_place_id:
        st.warning("Please enter and select a valid address from the dropdown suggestions for both locations.")
    else:
        try:
            day_type = "Weekday" if pickup_date.weekday() < 5 else "Weekend"
            time_of_day = get_time_of_day(pickup_time.hour)
            departure_time = datetime.combine(pickup_date, pickup_time)

            with st.spinner("Calculating distance..."):
                matrix_result = gmaps.distance_matrix(
                    origins=[f"place_id:{start_place_id}"],
                    destinations=[f"place_id:{end_place_id}"],
                    mode="driving", departure_time=departure_time
                )
            element = matrix_result['rows'][0]['elements'][0]

            if element['status'] == 'OK':
                trip_distance_km = element['distance']['value'] / 1000.0
                payload = {
                    "Trip_Distance_km": trip_distance_km, "Time_of_Day": time_of_day,
                    "Day_of_Week": day_type, "Passenger_Count": passenger_count,
                    "Traffic_Conditions": traffic_conditions, "Weather": weather
                }

                # --- LIVE API CALL ---
                st.info("Sending data to the prediction API...")
                response = post_api_endpoint(data=payload)
                response.raise_for_status()  # Will raise an error for bad responses (4xx or 5xx)

                response_data = response.json()
                # Ensure you use the correct key that your API returns, e.g., "predicted_fare"
                predicted_fare = response_data

                if predicted_fare is not None:
                    st.success(f"## 💵 Predicted Fare: {predicted_fare:.2f} Dollar")
                else:
                    st.error("Prediction API returned a response, but it did not contain a valid fare key.")
            else:
                st.error(f"Could not find a route. Google Maps says: {element['status'].replace('_', ' ').title()}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the prediction API. Please ensure it's running. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

