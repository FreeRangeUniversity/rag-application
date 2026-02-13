import os
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_openai import OpenAIEmbeddings

from config import COLLECTION_NAME, QDRANT_PATH, EMBED_MODEL
from pdf_loader import load_pdf
from splitter import splitter

DOCS_PATH = "data/documents"
BATCH_SIZE = 50  # Number of chunks per embedding request
MAX_RETRIES = 5  # Retry on rate limits
SLEEP_BETWEEN_RETRIES = 5  # seconds

# Initialize Qdrant client and embeddings
client = QdrantClient(path=QDRANT_PATH)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# Create collection if it doesn't exist
existing_collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    print(f"Creating collection {COLLECTION_NAME}")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,  # text-embedding-3-large
            distance=Distance.COSINE
        )
    )
else:
    print(f"Using existing collection {COLLECTION_NAME}")

# Helper: embedding with retry
def embed_with_retry(chunks):
    for attempt in range(MAX_RETRIES):
        try:
            return embeddings.embed_documents(chunks)
        except Exception as e:  # fallback if OpenAIError import fails
            print(f"Error during embedding: {e}")
            print(f"Retrying in {SLEEP_BETWEEN_RETRIES}s... (Attempt {attempt+1})")
            time.sleep(SLEEP_BETWEEN_RETRIES)
    raise Exception("Failed to embed after multiple retries")


# Gather all chunks and metadata
all_chunks = []
chunk_meta = []

for filename in os.listdir(DOCS_PATH):
    if not filename.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(DOCS_PATH, filename)
    pages = load_pdf(file_path)
    print(f"Loaded {len(pages)} pages from {filename}")

    for page in pages:
        chunks = [
            c for c in splitter.split_text(page["text"])
            if len(c.strip()) > 50
        ]

        all_chunks.extend(chunks)
        chunk_meta.extend([{"source": filename, "page": page["page"]}] * len(chunks))

print(f"Total chunks to embed: {len(all_chunks)}")

# Embed in batches and upsert to Qdrant
points = []
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch_chunks = all_chunks[i:i+BATCH_SIZE]
    batch_meta = chunk_meta[i:i+BATCH_SIZE]

    vectors = embed_with_retry(batch_chunks)

    for text, vector, meta in zip(batch_chunks, vectors, batch_meta):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": text,
                    "source": meta["source"],
                    "page": meta["page"]
                }
            )
        )

    # Upsert each batch to avoid holding everything in memory
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted batch {i//BATCH_SIZE + 1} ({len(points)} points so far)")
    points = []  # Clear points after upsert

print("✅ All chunks ingested successfully!")
client.close()
