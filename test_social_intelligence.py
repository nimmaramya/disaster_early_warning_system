from utils.social_intelligence import build_social_text

text, score = build_social_text("Hyderabad")

print("News Text:")
print(text)

print("\nKeyword Score:", score)