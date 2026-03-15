from utils.dynamic_fusion import dynamic_fusion

weather = 0.7
satellite = 0.3
social = 0.8

risk, weights = dynamic_fusion(weather, satellite, social)

print("Final Risk:", risk)
print("Weights:", weights)