import googlemaps
import streamlit as st
# DISCLAIMER: Ive used a fair bit of LLM help to write this code, especially for the Google Maps API parts.
# I understand the concepts and logic, but LLMs help me with syntax and API details.
class GoogleMapsClient:
    """A wrapper for Google Maps API calls to handle geocoding and distance calculations."""

    def __init__(self, api_key):
        """Initializes the Google Maps client."""
        if not api_key:
            raise ValueError("Google Maps API key is required.")
        self.gmaps = googlemaps.Client(key=api_key)

    # Use Streamlit's caching to avoid repeated API calls for the same input
    @st.cache_data(show_spinner=False)
    def get_place_suggestions(_self, input_text, session_token, country_code="SE"):
        """
        Fetches place autocomplete suggestions from Google Maps API.
        Caches the results to prevent redundant calls for the same text.
        
        Args:
            _self: The instance of the class (st.cache_data requires the first arg to be the instance).
            input_text (str): The partial address typed by the user.
            session_token: A unique token for the user's session to group requests for billing.
            country_code (str): The country code to bias results (e.g., "SE" for Sweden).

        Returns:
            list: A list of dictionaries, each containing a 'description' and 'place_id'.
        """
        if not input_text:
            return []
        try:
            # I add a country component restriction to bias results towards Sweden
            suggestions = _self.gmaps.places_autocomplete(
                input_text,
                session_token=session_token,
                components={'country': country_code}
            )
            # Format the suggestions for easy use in the frontend
            return [
                {"description": s['description'], "place_id": s['place_id']}
                for s in suggestions
            ]
        except Exception as e:
            st.error(f"Error fetching suggestions: {e}")
            return []

    # Use caching for distance calculation as well
    @st.cache_data(show_spinner="Calculating road distance...")
    def get_road_distance(_self, origin_place_id, destination_place_id):
        """
        Calculates the driving distance between two locations using their Place IDs.

        Args:
            _self: The instance of the class.
            origin_place_id (str): The Google Maps Place ID for the origin.
            destination_place_id (str): The Google Maps Place ID for the destination.

        Returns:
            dict: A dictionary containing the distance text (e.g., "15.5 km") and
                  distance in meters, or None if an error occurs.
        """
        try:
            # Use Place IDs for accuracy: "place_id:ChIJ..."
            result = _self.gmaps.distance_matrix(
                origins=[f"place_id:{origin_place_id}"],
                destinations=[f"place_id:{destination_place_id}"],
                mode="driving"
            )

            # Check if the API returned a valid route
            if result['rows'][0]['elements'][0]['status'] == 'OK':
                distance_text = result['rows'][0]['elements'][0]['distance']['text']
                distance_meters = result['rows'][0]['elements'][0]['distance']['value']
                return {"text": distance_text, "meters": distance_meters}
            else:
                st.warning("Could not calculate the distance for the selected route.")
                return None
        except Exception as e:
            st.error(f"Error calculating distance: {e}")
            return None