from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the model we want to use
model_name = "distilbert/distilgpt2"
revision_id = "2290a62682d06624634c1f46a6ad5be0f47f38aa"

# Load the tokenizer and model
print(f"Downloading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision_id)

print(f"Downloading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision_id)


# Print model information
print(f"\nModel loaded: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")


# Test the model with a sample input
test_text = "Artificial intelligence is"
input_ids = tokenizer(test_text, return_tensors="pt").input_ids

print(f"\nGenerating sample completion for: '{test_text}'")
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Sample output: '{generated_text}'")

print("\nModel download and verification complete.")



