from utils.weather_predict import predict_weather_risk

city = "Hyderabad"

risk = predict_weather_risk(city)

print("Weather Risk Score:", risk)