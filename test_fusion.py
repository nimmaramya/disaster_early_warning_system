from utils.fusion import fuse_risk, risk_level

weather = 0.20
cnn = 0.70
nlp = 0.60

final_score = fuse_risk(weather, cnn, nlp)

print("Final Risk Score:", final_score)
print("Risk Level:", risk_level(final_score))