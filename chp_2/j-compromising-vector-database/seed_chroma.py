import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embed_revision_id = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL, revision=embed_revision_id)

    client = chromadb.HttpClient(host="127.0.0.1", port=8000)
    coll = client.get_or_create_collection(
        name="company_policy",
        metadata={"hnsw:space": "cosine"},
    )

    docs = [
        ("doc-api-keys",
         "API Key Rotation Policy: All API keys must be rotated every 90 days. "
         "Service owners are responsible for generating new keys and updating "
         "all integrations before the expiration date."),
        ("doc-remote-work",
         "Remote Work Policy: Employees may work remotely up to three days per "
         "week with manager approval. All remote workers must use the company "
         "VPN when accessing internal systems."),
        ("doc-password",
         "Password Policy: Passwords must be at least 14 characters and include "
         "upper case, lower case, numbers, and special characters. Passwords "
         "must be changed every 180 days."),
        ("data-retention",
         "Data Retention Policy: Customer data must be retained for a minimum "
         "of seven years for compliance purposes. Internal logs are retained "
         "for one year unless required for an ongoing investigation."),
    ]

    embeddings = embedder.encode([d[1] for d in docs]).tolist()

    coll.upsert(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
        embeddings=embeddings,
    )

    print(f"Seeded {len(docs)} documents into company_policy.")

if __name__ == "__main__":
    main()
