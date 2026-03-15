from utils.cnn_predict import predict_cnn_risk

image_path = "test_image.jpg"

risk = predict_cnn_risk(image_path)

print("CNN Risk Score:", risk)