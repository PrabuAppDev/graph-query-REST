from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import PointStruct
import logging
from query_engine import graph  # Import the graph from your query_engine.py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Initialize vectorizer
vectorizer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def setup_vector_db(graph):
    """
    Create and populate a persistent vector database with graph data.

    Args:
        graph (dict): Graph representation with `nodes` and `edges`.

    Returns:
        QdrantClient: Persistent Qdrant client instance with indexed data.
    """
    try:
        # Connect to persistent Qdrant database
        client = qdrant_client.QdrantClient(url="http://localhost:6333")  # Persistent DB

        # Ensure the collection exists or create it
        if client.collection_exists("systems"):
            logging.info("Collection 'systems' already exists. Skipping recreation.")
        else:
            client.create_collection(
                collection_name="systems",
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=384, distance="Cosine"  # Embedding size from the model
                )
            )

        # Populate the collection with data
        points = []
        for edge in graph["edges"]:
            text = f"{edge['source']} interacts with {edge['target']} via {edge['integration']}"
            embedding = vectorizer.encode(text).tolist()
            points.append(
                PointStruct(
                    id=len(points),
                    vector=embedding,
                    payload={
                        "source": edge["source"],
                        "target": edge["target"],
                        "integration": edge["integration"],
                        "context": edge["context"],
                    }
                )
            )

        client.upsert(collection_name="systems", points=points)
        logging.info(f"Vector database populated with {len(points)} points!")
        return client

    except Exception as e:
        logging.error(f"Error setting up vector database: {e}")
        return None

if __name__ == "__main__":
    client = setup_vector_db(graph)
    if client:
        logging.info("Vector database service is running.")
    else:
        logging.error("Vector database service failed to start.")
