import chromadb

def main():
    print("Connecting to Chroma (no credentials)...\n")
    client = chromadb.HttpClient(host="127.0.0.1", port=8000)

    # List all collections
    collections = client.list_collections()
    print(f"Collections found: {len(collections)}")
    for name in collections:
        print(f"  - {name}")
    print()

    # Dump company_policy
    coll = client.get_collection("company_policy")
    data = coll.get(include=["documents", "metadatas"])

    print(f"Documents in company_policy: {len(data['ids'])}\n")
    for i in range(len(data["ids"])):
        print(f"ID: {data['ids'][i]}")
        print(f"Text: {data['documents'][i]}")
        print()

if __name__ == "__main__":
    main()
