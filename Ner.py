from transformers import pipeline

# Load BERT model fine-tuned for NER
ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# Sample text
text = """
Sundar Pichai is the CEO of Google. He was born in India and recently visited New York.
"""

# Perform NER
results = ner(text)

print("Named Entities:\n")

for entity in results:
    print(f"Entity : {entity['word']}")
    print(f"Type   : {entity['entity_group']}")
    print(f"Score  : {entity['score']:.4f}")
    print("-" * 30)