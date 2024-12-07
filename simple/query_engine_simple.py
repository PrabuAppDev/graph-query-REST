import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

vectorizer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def setup_vector_db(edges, drop_existing=True):
    client = QdrantClient(path=":memory:")
    if drop_existing and client.collection_exists("systems"):
        client.delete_collection(collection_name="systems")

    client.create_collection(
        collection_name="systems",
        vectors_config=VectorParams(size=384, distance="Cosine")
    )

    populate_vector_db(client, edges)
    return client

def populate_vector_db(client, edges):
    points = []
    for idx, edge in enumerate(edges):
        description = f"{edge['consumer']} interacts with {edge['producer']} via {edge['integration']}"
        vector = vectorizer.encode(description).tolist()
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "consumer": edge["consumer"],
                    "producer": edge["producer"],
                    "integration": edge["integration"]
                }
            )
        )
    client.upsert(collection_name="systems", points=points)

def retrieve_context(client, query):
    vector = vectorizer.encode(query).tolist()
    results = client.search(collection_name="systems", query_vector=vector, limit=5)
    context = [result.payload for result in results]

    edges = [{
        "consumer": item["consumer"],
        "producer": item["producer"],
        "integration": item["integration"]
    } for item in context]

    return context, edges