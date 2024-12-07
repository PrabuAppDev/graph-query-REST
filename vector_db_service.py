from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize vectorizer
vectorizer = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def setup_vector_db(edges, drop_existing=True):
    """
    Set up and populate the Qdrant vector database.

    Args:
        edges (list): List of edges with 'consumer' and 'producer'.
        drop_existing (bool): Whether to drop the existing collection.

    Returns:
        QdrantClient: Initialized Qdrant client.
    """
    try:
        client = QdrantClient(path=":memory:")
        logging.info("Qdrant initialized in memory.")

        # Drop the collection if it exists and `drop_existing` is True
        if drop_existing and client.collection_exists("systems"):
            client.delete_collection(collection_name="systems")
            logging.info("Existing 'systems' collection dropped.")

        # Recreate the collection
        client.recreate_collection(
            collection_name="systems",
            vectors_config=VectorParams(size=384, distance="Cosine")
        )
        logging.info("Collection 'systems' created.")

        # Populate the collection
        populate_vector_db(client, edges)

        return client

    except Exception as e:
        logging.error(f"Error setting up vector database: {e}")
        return None


def populate_vector_db(client, edges):
    """
    Populate the Qdrant vector database with edges data.
    """
    try:
        points = []
        for idx, edge in enumerate(edges):
            # Construct the description without context
            description = f"{edge['consumer']} interacts with {edge['producer']} via {edge['integration']}"
            vector = vectorizer.encode(description).tolist()

            # Add points without including the 'context' field
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
        logging.info(f"Added {len(points)} edges to the vector database.")
        
    except Exception as e:
        logging.error(f"Error populating vector database: {e}")


def retrieve_context(client, query):
    """
    Query the Qdrant vector database.
    """
    try:
        vector = vectorizer.encode(query).tolist()
        results = client.search(
            collection_name="systems",
            query_vector=vector,
            limit=5,
        )
        return [result.payload for result in results]

    except Exception as e:
        logging.error(f"Error retrieving context: {e}")
        return []


if __name__ == "__main__":
    # Example edges without the need for 'context'
    edges = [
        {"consumer": "System A", "producer": "System B", "integration": "API"},
        {"consumer": "System B", "producer": "System C", "integration": "Database"},
        {"consumer": "System A", "producer": "System D", "integration": "File Transfer"},
    ]

    # Set up the database
    client = setup_vector_db(edges)  # Pass `edges` as an argument here
    if client:
        logging.info("Vector database populated.")

        # Test querying
        query = "What systems interact with System A?"
        context = retrieve_context(client, query)
        logging.info(f"Retrieved Context for query '{query}': {context}")
    else:
        logging.error("Vector database service failed to start.")
