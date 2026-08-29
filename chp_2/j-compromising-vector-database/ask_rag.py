import readline  # enables arrow keys / line editing for input()

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
revision_id = "0a67737cc96d2554230f90338b163bc6380a2a85"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embed_revision_id = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SYSTEM = (
    "You are a company policy assistant. Answer the user's question "
    "based only on the retrieved context. If the context does not "
    "contain the answer, say that you do not have information on that topic."
)
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"
SEP = "-" * 60


def retrieve(query, collection, embedder, k=3):
    q_vec = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=q_vec, n_results=k)
    return results


def generate(model, tokenizer, device, context, query):
    prompt_text = (
        f"Context:\n{context}\n\nQuestion: {query}\n\n"
        f"Answer based on the context provided:"
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL, revision=embed_revision_id)

    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision_id)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision_id,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)

    client = chromadb.HttpClient(host="127.0.0.1", port=8000)
    collection = client.get_collection("company_policy")

    print("Ready. Enter a question, or type x / quit / exit (or Ctrl+C) to stop.\n")

    while True:
        try:
            question = input(f"{GREEN}You>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"x", "quit", "exit"}:
            print("Bye.")
            break

        results = retrieve(question, collection, embedder)
        context_parts = []
        for i, doc in enumerate(results["documents"][0]):
            doc_id = results["ids"][0][i]
            context_parts.append(f"[{doc_id}] {doc}")
        context = "\n\n".join(context_parts)

        print(f"\nRetrieved {len(context_parts)} chunk(s).")
        for part in context_parts:
            snippet = part[:100]
            print(f"  {snippet}...")
        print()

        answer = generate(model, tokenizer, device, context, question)
        print(f"{BLUE}Assistant>{RESET}", answer)
        print(SEP)


if __name__ == "__main__":
    main()
