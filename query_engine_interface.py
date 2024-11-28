import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import PointStruct
from openai import OpenAI

__all__ = [
    "generate_json_with_openai_few_shot",
    "retrieve_context_from_vector_db",
    "setup_vector_db",
    "construct_graph",
]

# -------------------- Load CSV and Construct Graph --------------------

def load_csv(file_path):
    """Load a CSV file and return a DataFrame."""
    try:
        data = pd.read_csv(file_path)
        print("CSV loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def construct_graph(data):
    """Construct a graph representation from CSV data."""
    nodes = list(set(data['Consumer']).union(set(data['Producer'])))
    edges = data.rename(
        columns={
            "Consumer": "consumer",
            "Producer": "producer",
            "Integration Type": "integration",
            "Context-Domain": "context",
        }
    ).to_dict(orient="records")
    return {"nodes": [{"id": node} for node in nodes], "edges": edges}


# -------------------- Vector Database Setup --------------------

vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def setup_vector_db(graph):
    """Create and populate an in-memory vector database with graph data."""
    try:
        client = qdrant_client.QdrantClient(":memory:")
        if client.collection_exists("systems"):
            client.delete_collection("systems")
        client.create_collection(
            collection_name="systems",
            vectors_config=qdrant_client.http.models.VectorParams(size=384, distance="Cosine")
        )

        points = []
        for edge in graph["edges"]:
            text = f"{edge['consumer']} interacts with {edge['producer']} via {edge['integration']}"
            embedding = vectorizer.encode(text).tolist()
            points.append(
                PointStruct(
                    id=len(points),
                    vector=embedding,
                    payload=edge,
                )
            )

        client.upsert(collection_name="systems", points=points)
        print(f"Vector database populated with {len(points)} points!")
        return client
    except Exception as e:
        print(f"Error setting up vector database: {e}")
        return None


# -------------------- Retrieve Context --------------------

def retrieve_context_from_vector_db(client, query, top_k=5):
    """Retrieve relevant context for the query from the vector database."""
    try:
        query_embedding = vectorizer.encode(query).tolist()
        results = client.search(
            collection_name="systems",
            query_vector=query_embedding,
            limit=top_k
        )
        return [result.payload for result in results]
    except Exception as e:
        print(f"Error querying vector database: {e}")
        return []


# -------------------- OpenAI Integration --------------------

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_response(messages, model="gpt-4o-mini", max_tokens=500, temperature=0.7):
    """Generate a response using OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def generate_json_with_openai_few_shot(context, query):
    """Generate a JSON response using OpenAI API with few-shot examples."""
    try:
        formatted_context = "\n".join([
            f"Consumer: {item['consumer']}, Producer: {item['producer']}, Integration: {item['integration']}, Context: {item['context']}"
            for item in context
        ])
        few_shot_examples = """
Example 1:
Context:
Consumer: System A, Producer: System B, Integration: API, Context: Authentication

Question: What are the systems that interact with System A?

JSON Response:
{
  "query": "What are the systems that interact with System A?",
  "nodes": [
    {"id": "System A"},
    {"id": "System B"}
  ],
  "edges": [
    {"consumer": "System A", "producer": "System B", "integration": "API", "context": "Authentication"}
  ]
}
"""
        messages = [
            {"role": "system", "content": "You are an assistant that generates JSON responses."},
            {"role": "user", "content": few_shot_examples},
            {"role": "user", "content": f"Context: {formatted_context}\nQuestion: {query}\nJSON Response:"},
        ]
        return generate_response(messages, model="gpt-4o-mini", max_tokens=500, temperature=0.0)
    except Exception as e:
        print(f"Error generating JSON response with OpenAI: {e}")
        return None


# -------------------- Main Function for Testing --------------------

if __name__ == "__main__":
    # Load CSV
    file_path = "sample_integration_data.csv"
    integration_data = load_csv(file_path)
    if integration_data is None:
        exit("Failed to load CSV.")

    # Construct graph
    graph = construct_graph(integration_data)
    print("Graph Representation:")
    print("Nodes:", graph["nodes"])
    print("Edges:", graph["edges"])

    # Set up vector database
    vector_db_client = setup_vector_db(graph)
    if not vector_db_client:
        exit("Failed to set up vector database.")

    # Test query
    test_query = "What are the systems that interact with Finance System?"
    retrieved_context = retrieve_context_from_vector_db(vector_db_client, test_query)
    print("Retrieved Context:", retrieved_context)

    if retrieved_context:
        # Generate JSON response
        json_response = generate_json_with_openai_few_shot(retrieved_context, test_query)
        if json_response:
            print("Generated JSON Response:", json_response)
        else:
            print("Failed to generate JSON response.")
    else:
        print("No context retrieved from vector database.")