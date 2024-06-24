import streamlit as st
import requests
import json
import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Initialize the Mistral client
mistral_api_key = os.getenv('MISTRAL_API_KEY')
client = MistralClient(api_key=mistral_api_key)

def get_coordinates(location):
    # Check if the location is a zip code (assuming 5-digit US zip codes)

    if location.isdigit() and len(location) == 5:
        # Use Zippopotam.us for zip code lookup
        zip_url = f"https://api.zippopotam.us/us/{location}"
        zip_response = requests.get(zip_url)
        
        if zip_response.status_code != 200:
            return None, "Unable to find location"
        
        zip_data = zip_response.json()
        # print('zip data:', zip_data)
        
        if not zip_data:
            return None, "Location not found"
        
        lat = float(zip_data["places"][0]["latitude"])
        lon = float(zip_data["places"][0]["longitude"])

        # Extract formatted location
        formatted_location = zip_data["places"][0]["place name"]+', '+zip_data["places"][0]["state"]

    else:
        # Use Nominatim for geocoding
        nominatim_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
        headers = {
            "User-Agent": "WeatherApp/1.0"  # It's good practice to identify your application
        }
        geo_response = requests.get(nominatim_url, headers=headers)
        
        if geo_response.status_code != 200:
            return None, "Unable to find location"
        
        geo_data = geo_response.json()
        # print('geo data:', geo_data)
        
        if not geo_data:
            return None, "Location not found"
        
        result = geo_data[0]
        lat = float(result["lat"])
        lon = float(result["lon"])
        
        # Extract formatted location
        formatted_location = result.get("display_name", location)

    return (lat, lon, formatted_location), None


def get_weather(location, use_fahrenheit):
    coordinates, error = get_coordinates(location)
    if error:
        return {"error": error}
    
    lat, lon, formatted_location = coordinates
    
    # Weather API call
    temperature_unit = "fahrenheit" if use_fahrenheit else "celsius"
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&temperature_unit={temperature_unit}&forecast_days=1"
    weather_response = requests.get(weather_url)
    
    if weather_response.status_code != 200:
        return {"error": "Unable to fetch weather data"}
    
    weather_data = weather_response.json()
    # print('weather_data:', weather_data)
    
    # Convert weather code to description
    weather_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    
    weather_code = weather_data["current"]["weather_code"]
    weather_description = weather_codes.get(weather_code, "Unknown")
    
    return {
        "temperature": weather_data["current"]["temperature_2m"],
        "humidity": weather_data["current"]["relative_humidity_2m"],
        "condition": weather_description,
        "unit": "°F" if use_fahrenheit else "°C",
        "location": formatted_location
    }


# Define the function schema
weather_function = {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name or ZIP code, e.g. New York or 10001"
            }
        },
        "required": ["location", "use_fahrenheit"],
      }
    }
}

# Streamlit app
st.title("Weather App with Mistral AI")

# User input
location = st.text_input("Enter a location (zip code or some combination of city/state/country):")

# Temperature unit toggle
use_fahrenheit = st.toggle("Use Fahrenheit", value=False)

if st.button("Get Weather"):
    if location:
        with st.spinner("Fetching weather information..."):
            # Set up the conversation
            messages = [
                ChatMessage(role="user", content=f"What's the weather like in {location}? Please provide the temperature in {'Fahrenheit' if use_fahrenheit else 'Celsius'}.")
            ]

            # Make the API call to Mistral
            response = client.chat(
                model="mistral-large-latest",
                messages=messages,
                tools=[weather_function],
                tool_choice="auto"
            )

            # Add response
            messages.append(response.choices[0].message)

            # Extract the function call
            function_call = response.choices[0].message.tool_calls[0].function

            # If a function was called, execute it and generate a response
            if function_call:
                function_name = function_call.name
                function_args = json.loads(function_call.arguments)

                if function_name == "get_weather":
                    weather_data = get_weather(function_args["location"], use_fahrenheit)
                    
                    if "error" in weather_data:
                        st.error(f"Error: {weather_data['error']}")
                    else:
                        # Generate a response using the weather data
                        response = client.chat(
                            model="mistral-large-latest",
                            messages=messages + [
                                ChatMessage(role="tool", content=json.dumps(weather_data), name="get_weather")
                            ]
                        )

                        st.success(response.choices[0].message.content)

                        # Display raw weather data
                        st.subheader("Raw Weather Data")
                        st.json(weather_data)
            else:
                st.warning("Unable to process the weather request.")
    else:
        st.warning("Please enter a location.")