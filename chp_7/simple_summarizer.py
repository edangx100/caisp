from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
revision_id = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision_id)

# Create a pipeline for text generation using the model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, max_new_tokens=250, do_sample=False)

def summarize_text(text: str) -> str:
    # Define the prompt with the text to summarize
    prompt = f"Summarize the following: {text}"

    # Generate the summary using the model
    #summary = pipe(prompt, max_length=100, num_return_sequences=1)
    summary = pipe(prompt)

    # Return the generated summary (the model output)
    return summary[0]['generated_text']

# Test the summarizer function
if __name__ == "__main__":
    sample_text = """
    Around 85% of people infected with the chikungunya virus experience symptoms, typically beginning with a sudden high fever above 39 °C (102 °F).[17] The fever is soon followed by severe muscle and joint pain.[18] Pain usually affects multiple joints in the arms and legs, and is symmetric – i.e. if one elbow is affected, the other is as well.[18] People with chikungunya also frequently experience headache, back pain, nausea, and fatigue.[18] Around half of those affected develop a rash, with reddening and sometimes small bumps on the palms, foot soles, torso, and face.[18] For some, the rash remains constrained to a small part of the body; for others, the rash can be extensive, covering more than 90% of the skin.[17] Some people experience gastrointestinal issues, with abdominal pain and vomiting. Others experience eye problems, namely sensitivity to light, conjunctivitis, and pain behind the eye.[18] This first set of symptoms – called the "acute phase" of chikungunya – lasts around a week, after which most symptoms resolve on their own.[18]

	Many people continue to have symptoms after the "acute phase" resolves, termed the "post-acute phase" for symptoms lasting three weeks to three months, and the "chronic stage" for symptoms lasting longer than three months.[18] In both cases, the lasting symptoms tend to be joint pains: arthritis, tenosynovitis, and/or bursitis.[18] If the affected person has pre-existing joint issues, these tend to worsen.[18] Overuse of a joint can result in painful swelling, stiffness, nerve damage, and neuropathic pain.[18] Typically the joint pain improves with time; however, the chronic stage can last anywhere from a few months to several years.[18]

	Joint pain is reported in 87–98% of cases, and nearly always occurs in more than one joint, though joint swelling is uncommon.[19] Typically the affected joints are located in both arms and legs. Joints are more likely to be affected if they have previously been damaged by disorders such as arthritis.[20] Pain most commonly occurs in peripheral joints, such as the wrists, ankles, and joints of the hands and feet as well as some of the larger joints, typically the shoulders, elbows and knees.[19][20] Pain may also occur in the muscles or ligaments. In more than half of cases, normal activity is limited by significant fatigue and pain.[19] Infrequently, inflammation of the eyes may occur in the form of iridocyclitis, or uveitis, and retinal lesions may occur.[21] Temporary damage to the liver may occur.[22]
    """

    # Call the summarize_text function
    summary = summarize_text(sample_text)
    print("-"*50)
    print(summary)
    print("-"*50)
