from utils.nlp_predict import predict_text_risk

text = "Severe flooding in the city. People are being evacuated."

risk = predict_text_risk(text)

print("NLP Risk Score:", risk)