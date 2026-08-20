import os
import json
import chromadb
from sentence_transformers import SentenceTransformer


chroma = chromadb.PersistentClient(path="./chroma_db")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "startup_knowledge"
BATCH_SIZE = 64

def embed_text(text):
    return embed_model.encode(text).tolist()

def reset_collection():
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return chroma.get_or_create_collection(COLLECTION_NAME)

def format_list(title, values):
    if not values:
        return f"{title}: Not available"
    if isinstance(values, list):
        return f"{title}: " + " | ".join(str(value) for value in values)
    return f"{title}: {values}"

def decision_record_to_document(record):
    return "\n".join([
        f"Decision ID: {record.get('Decision_ID', 'Unknown')}",
        f"Domain: {record.get('Domain', 'Unknown')}",
        f"Company: {record.get('Company', 'Unknown')}",
        f"Year: {record.get('Year', 'Unknown')}",
        f"Company Stage: {record.get('Company_Stage', 'Unknown')}",
        f"Decision Category: {record.get('Decision_Category', 'Unknown')}",
        f"Problem Statement: {record.get('Problem_Statement', '')}",
        f"Business Goal: {record.get('Business_Goal', '')}",
        f"Market Condition: {record.get('Market_Condition', '')}",
        f"Economic Environment: {record.get('Economic_Environment', '')}",
        f"Competitor Situation: {record.get('Competitor_Situation', '')}",
        f"Customer Segment: {record.get('Customer_Segment', '')}",
        f"Budget Available: {record.get('Budget_Available', '')}",
        f"Team Size: {record.get('Team_Size', '')}",
        f"Revenue Before Decision: {record.get('Revenue_Before_Decision', '')}",
        f"Burn Rate: {record.get('Burn_Rate', '')}",
        f"Cash Runway: {record.get('Cash_Runway', '')} months",
        f"Decision Taken: {record.get('Decision_Taken', '')}",
        f"Reason for Decision: {record.get('Reason_for_Decision', '')}",
        format_list("Arguments in Favor", record.get("Arguments_in_Favor")),
        format_list("Arguments Against", record.get("Arguments_Against")),
        f"Outcome: {record.get('Outcome', '')}",
        f"Lessons Learned: {record.get('Lessons_Learned', '')}",
        f"Evidence Level: {record.get('Evidence_Level', '')}",
        f"Source: {record.get('Source', '')}",
    ])

def record_metadata(record, filename):
    return {
        "filename": filename,
        "decision_id": str(record.get("Decision_ID", "")),
        "domain": str(record.get("Domain", "")),
        "company": str(record.get("Company", "")),
        "year": str(record.get("Year", "")),
        "company_stage": str(record.get("Company_Stage", "")),
        "decision_category": str(record.get("Decision_Category", "")),
        "outcome": str(record.get("Outcome", "")),
        "evidence_level": str(record.get("Evidence_Level", "")),
    }

def add_batch(collection, documents, ids, metadatas):
    embeddings = embed_model.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

def add_json_file(collection, filepath, filename, start_index):
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    if isinstance(records, dict):
        records = [records]

    documents = []
    ids = []
    metadatas = []
    added = 0

    for offset, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        decision_id = record.get("Decision_ID") or f"{start_index + offset}"
        documents.append(decision_record_to_document(record))
        ids.append(f"decision_{decision_id}")
        metadatas.append(record_metadata(record, filename))

        if len(documents) >= BATCH_SIZE:
            add_batch(collection, documents, ids, metadatas)
            added += len(documents)
            print(f"Added {added} records from {filename}")
            documents, ids, metadatas = [], [], []

    if documents:
        add_batch(collection, documents, ids, metadatas)
        added += len(documents)

    print(f"Added: {filename} ({added} records)")
    return added

def add_text_file(collection, filepath, filename, doc_index):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    vector = embed_text(text)

    collection.add(
        documents=[text],
        embeddings=[vector],
        ids=[f"doc_{doc_index}"],
        metadatas=[{"filename": filename, "source_type": "text"}]
    )
    print(f"Added: {filename}")
    return 1

def build_db():
    folder = "./knowledge_base"
    files = sorted(os.listdir(folder))
    collection = reset_collection()
    
    added = 0
    for filename in files:
        filepath = os.path.join(folder, filename)
        if filename.endswith(".json"):
            added += add_json_file(collection, filepath, filename, added)
        elif filename.endswith(".txt"):
            added += add_text_file(collection, filepath, filename, added)
    
    print(f"Database built with {added} documents")

build_db()
