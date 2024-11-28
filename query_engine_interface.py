import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import PointStruct
from openai import OpenAI

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
            "Context-Domain": "context"
        }
    ).to_dict(orient="records")

    graph = {
        "nodes": [{"id": node} for node in nodes],
        "edges": edges
    }
    return graph

# -------------------- Vector Database Setup --------------------

vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def setup_vector_db(graph):
    """Create and populate an in-memory vector database with graph data."""
    try:
        client = qdrant_client.QdrantClient(":memory:")
        
        # Delete existing collection if it exists
        if client.collection_exists("systems"):
            client.delete_collection("systems")
        
        # Create a new collection
        client.create_collection(
            collection_name="systems",
            vectors_config=qdrant_client.http.models.VectorParams(
                size=384, distance="Cosine"
            )
        )

        # Add data to the vector database
        points = []
        for edge in graph["edges"]:
            text = f"{edge['consumer']} interacts with {edge['producer']} via {edge['integration']}"
            embedding = vectorizer.encode(text).tolist()
            points.append(
                PointStruct(
                    id=len(points),
                    vector=embedding,
                    payload={
                        "consumer": edge["consumer"],
                        "producer": edge["producer"],
                        "integration": edge["integration"],
                        "context": edge["context"]
                    }
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
        print(f"Sending request to OpenAI with messages: {messages}")
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
    """Generate a strict JSON response using the OpenAI API with few-shot examples."""
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
            {"role": "user", "content": f"Context:\n{formatted_context}\nQuestion: {query}\nJSON Response:"}
        ]

        response_content = generate_response(messages, model="gpt-4o-mini", max_tokens=500, temperature=0.0)
        return response_content.strip()
    except Exception as e:
        print(f"Error generating JSON response with OpenAI: {e}")
        return None

# -------------------- Main Function --------------------

if __name__ == "__main__":
    # Load CSV
    file_path = 'sample_integration_data.csv'
    integration_data = load_csv(file_path)
    if integration_data is None:
        exit("Failed to load CSV.")

    # Construct Graph
    graph = construct_graph(integration_data)
    print("Graph Representation:")
    print("Nodes:", graph["nodes"])
    print("Edges:", graph["edges"])

    # Set Up Vector DB
    vector_db_client = setup_vector_db(graph)
    if not vector_db_client:
        exit("Failed to set up vector database.")

    # Query Vector DB
    query = "What are the systems that interact with Finance System?"
    retrieved_context = retrieve_context_from_vector_db(vector_db_client, query)
    if not retrieved_context:
        exit("No context retrieved from vector database.")

    # Generate JSON Response
    json_response = generate_json_with_openai_few_shot(retrieved_context, query)
    if json_response:
        print("Generated JSON Response:", json_response)
    else:
        print("Failed to generate JSON response.")