from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load PEGASUS model and tokenizer
model_name = "google/pegasus-xsum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """
Artificial Intelligence (AI) is one of the fastest-growing fields in computer science.
It enables machines to perform tasks that normally require human intelligence, such as
learning, reasoning, problem-solving, and decision-making. AI is widely used in
healthcare to diagnose diseases, in finance to detect fraud, in education to provide
personalized learning experiences, and in transportation to develop autonomous vehicles.
As AI continues to evolve, it is expected to improve productivity, reduce operational
costs, and create innovative solutions across various industries. However, ethical
concerns such as data privacy, bias, and transparency remain important challenges that
must be addressed.
"""

# Tokenize input (No "summarize:" prefix required)
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
    min_length=20,
    num_beams=4,
    early_stopping=True
)

# Decode summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Summary:")
print(summary)