#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Function to load and inspect the CSV
def load_csv(file_path):
    """
    Load a CSV file and return a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("CSV loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Path to the CSV file
file_path = 'sample_integration_data.csv'

# Load the CSV
integration_data = load_csv(file_path)

# Display the first few rows for inspection
if integration_data is not None:
    print("Sample Data:")
    display(integration_data.head())


# In[3]:


# Extract nodes and edges from the DataFrame
nodes = list(set(integration_data['Consumer']).union(set(integration_data['Producer'])))
edges = integration_data.rename(
    columns={
        "Consumer": "source",
        "Producer": "target",
        "Integration Type": "integration",
        "Context-Domain": "context"
    }
).to_dict(orient="records")

# Create the graph structure
graph = {
    "nodes": [{"id": node} for node in nodes],
    "edges": edges
}

# Print the graph for verification
print("Graph Representation:")
print("Nodes:", graph["nodes"])
print("Edges:", graph["edges"])


# In[ ]:


# !pip install -v qdrant_client


# In[5]:


from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import PointStruct

# Initialize vectorizer (e.g., SentenceTransformer)
vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to prepare and store data in the vector database
def setup_vector_db(graph):
    """
    Create and populate an in-memory vector database with graph data.

    Args:
        graph (dict): Graph representation with `nodes` and `edges`.

    Returns:
        QdrantClient: In-memory Qdrant client with indexed data.
    """
    try:
        client = qdrant_client.QdrantClient(":memory:")  # In-memory instance
        client.recreate_collection(
            collection_name="systems",
            vectors_config=qdrant_client.http.models.VectorParams(
                size=384, distance="Cosine"  # Embedding size from the model
            )
        )

        # Add graph data to the collection
        points = []
        for edge in graph["edges"]:
            text = f"{edge['source']} interacts with {edge['target']} via {edge['integration']}"
            embedding = vectorizer.encode(text).tolist()
            points.append(
                PointStruct(
                    id=len(points), vector=embedding,
                    payload={"source": edge["source"], "target": edge["target"],
                             "integration": edge["integration"], "context": edge["context"]}
                )
            )

        client.upsert(collection_name="systems", points=points)
        print(f"Vector database populated with {len(points)} points!")
        return client
    except Exception as e:
        print(f"Error setting up vector database: {e}")
        return None

# Set up the vector database
vector_db_client = setup_vector_db(graph)

# Confirm successful setup
if vector_db_client:
    print("Vector database setup successful!")


# In[7]:


def retrieve_context_from_vector_db(client, query, top_k=5):
    """
    Retrieve relevant context for the query from the vector database.

    Args:
        client (QdrantClient): Qdrant client instance.
        query (str): The natural language query.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Retrieved context from the vector database.
    """
    try:
        query_embedding = vectorizer.encode(query).tolist()
        results = client.search(
            collection_name="systems",
            query_vector=query_embedding,
            limit=top_k
        )
        context = [
            result.payload for result in results
        ]
        print(f"Retrieved {len(context)} context items for query: '{query}'")
        return context
    except Exception as e:
        print(f"Error querying vector database: {e}")
        return []

# Example query
example_query = "What are the systems that interact with Finance System?"
retrieved_context = retrieve_context_from_vector_db(vector_db_client, example_query)

# Confirm successful retrieval
if retrieved_context:
    print("Context retrieval successful!")
    print("Retrieved Context:", retrieved_context)
else:
    print("No context retrieved.")


# In[9]:


import os
import openai

# Initialize the OpenAI client with API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

def generate_response(messages, model="gpt-4o-mini", max_tokens=500, temperature=0.7):
    """
    Generate a response using the OpenAI API.

    Args:
        messages (list): List of messages for the conversation.
        model (str): Model to use for generation (e.g., "gpt-4o-mini").
        max_tokens (int): Maximum tokens for the output.
        temperature (float): Sampling temperature for randomness.

    Returns:
        str: Generated response content.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]

def generate_json_with_openai_few_shot(context, query):
    """
    Generate a strict JSON response using the OpenAI API with few-shot examples.

    Args:
        context (list): Context retrieved from the vector database.
        query (str): User query.

    Returns:
        str: JSON response as a string.
    """
    try:
        # Format context for the API
        formatted_context = "\n".join([
            f"Source: {item['source']}, Target: {item['target']}, Integration: {item['integration']}, Context: {item['context']}"
            for item in context
        ])

        # Few-shot examples to guide the model
        few_shot_examples = """
Example 1:
Context:
Source: System A, Target: System B, Integration: API, Context: Authentication
Source: System A, Target: System C, Integration: Webhook, Context: Notifications

Question: What are the systems that interact with System A?

JSON Response:
{
  "query": "What are the systems that interact with System A?",
  "nodes": [
    {"id": "System A"},
    {"id": "System B"},
    {"id": "System C"}
  ],
  "edges": [
    {"source": "System A", "target": "System B", "integration": "API", "context": "Authentication"},
    {"source": "System A", "target": "System C", "integration": "Webhook", "context": "Notifications"}
  ]
}

Example 2:
Context:
Source: Database X, Target: App Y, Integration: REST, Context: Data Transfer
Source: Database X, Target: App Z, Integration: GraphQL, Context: Analytics

Question: What are the systems that interact with Database X?

JSON Response:
{
  "query": "What are the systems that interact with Database X?",
  "nodes": [
    {"id": "Database X"},
    {"id": "App Y"},
    {"id": "App Z"}
  ],
  "edges": [
    {"source": "Database X", "target": "App Y", "integration": "REST", "context": "Data Transfer"},
    {"source": "Database X", "target": "App Z", "integration": "GraphQL", "context": "Analytics"}
  ]
}
"""

        # Combine examples with actual query
        messages = [
            {"role": "system", "content": "You are an assistant that generates JSON responses."},
            {"role": "user", "content": few_shot_examples},
            {
                "role": "user",
                "content": f"""
Now, based on the following context and question, generate ONLY JSON:

Context:
{formatted_context}

Question: {query}

JSON Response:
"""
            }
        ]

        # Generate the response using `generate_response`
        response_content = generate_response(messages, model="gpt-4o-mini", max_tokens=500, temperature=0.0)
        print("Raw OpenAI Response:", response_content)

        # Parse the JSON portion of the response
        start_index = response_content.find('{')
        end_index = response_content.rfind('}')
        if start_index != -1 and end_index != -1:
            response_json = response_content[start_index:end_index + 1]
            return response_json
        else:
            print("No valid JSON found in response.")
            return None

    except Exception as e:
        print(f"Error generating JSON response with OpenAI: {e}")
        return None


# In[11]:


# Import the functions from the saved Python file
from query_engine import generate_json_with_openai_few_shot, retrieve_context_from_vector_db

# Example query
example_query = "What are the systems that interact with Finance System?"

# Simulated vector DB retrieval
retrieved_context = retrieve_context_from_vector_db(None, example_query)
print("Retrieved Context:", retrieved_context)

# Test JSON generation
if retrieved_context:
    print("Testing JSON Generation with Imported Function...")
    json_response = generate_json_with_openai_few_shot(retrieved_context, example_query)

    # Print the generated JSON response
    if json_response:
        print("Generated JSON Response:", json_response)
    else:
        print("Failed to generate JSON response.")
else:
    print("No context retrieved from Vector DB.")

