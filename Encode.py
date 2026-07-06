from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="philschmid/MiniLM-L6-H384-uncased-sst2"
)

text = "The movie was absolutely amazing."

result = classifier(text)[0]

print(result)

label = result["label"]

# Handle both naming conventions
if label in ["POSITIVE", "LABEL_1"]:
    prediction = 1
else:
    prediction = 0

print("Prediction:", prediction)
print("Confidence:", result["score"])