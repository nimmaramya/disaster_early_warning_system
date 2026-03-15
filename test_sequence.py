from utils.weather_fetch import get_weather_data
from utils.weather_sequence import prepare_weather_sequence

city = "Hyderabad"

data = get_weather_data(city)

sequence = prepare_weather_sequence(data)

print("Sequence shape:", sequence.shape)
print(sequence)