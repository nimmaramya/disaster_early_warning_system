from utils.weather_fetch import get_weather_data

city = "Hyderabad"

data = get_weather_data(city)

print("Weather sequence:")
print(data)