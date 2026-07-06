from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1
)

result = generator(
    "Artificial Intelligence is ",
    max_new_tokens=100,
    pad_token_id=generator.tokenizer.eos_token_id,
    clean_up_tokenization_spaces=False
)

print(result[0]["generated_text"])