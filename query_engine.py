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
    print(integration_data.head())


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


import json
import logging

# Initialize logging
logger = logging.getLogger(__name__)

def create_prompt(query, context):
    """
    Create a prompt for AWS Bedrock to ensure a single JSON response.

    Args:
        query (str): User's query.
        context (list): List of interactions (source, target, integration, context).

    Returns:
        str: Formatted prompt for Bedrock.
    """
    # Log the context for debugging
    logging.info(f"Creating prompt with context: {context}")

    formatted_context = []
    for item in context:
        try:
            # Normalize keys to title case
            normalized_item = {
                "Source": item.get("source", ""),
                "Target": item.get("target", ""),
                "Integration": item.get("integration", ""),
                "Context": item.get("context", "")
            }
            formatted_context.append(
                f"Source: {normalized_item['Source']}, Target: {normalized_item['Target']}, "
                f"Integration: {normalized_item['Integration']}, Context: {normalized_item['Context']}"
            )
        except KeyError as e:
            logging.warning(f"Missing key in context item: {item}. Error: {e}")
            continue  # Skip items with missing keys

    # Join formatted context items
    formatted_context_str = "\n".join(formatted_context)

    prompt = f"""
<s>[INST] <<SYS>>
You are a structured data generation assistant. Your task is to generate a single cohesive JSON response based on the given context and query.

Rules:
1. The response must be a single JSON object with the following structure:
   {{
       "nodes": [
           {{"id": "System Name"}},
           ...
       ],
       "edges": [
           {{"source": "System A", "target": "System B", "integration": "Type", "context": "Details"}},
           ...
       ]
   }}
2. Do not split the JSON into multiple sections.
3. Do not include any text outside the JSON object.
4. Deduplicate nodes and edges in the response.

Query: What are the systems that interact with "{query}"?

Context:
{formatted_context_str}

Response:
</SYS> </INST>
"""
    logging.info(f"Generated prompt:\n{prompt}")
    return prompt


import json
import re
import logging

import json
import re
import logging

def extract_json_from_response(response):
    """
    Extract valid JSON from Bedrock's response, handling explanatory text.

    Args:
        response (dict): The response from AWS Bedrock.

    Returns:
        dict: Cleaned and deduplicated JSON object with nodes and edges.
    """
    try:
        # Get the raw response text from the "generation" field
        response_text = response.get("generation", "").strip()
        logging.info(f"Raw Bedrock Response Text:\n{response_text}\n")

        # Use regex to extract the JSON object
        json_match = re.search(r"\{(?:.|\n)*\}", response_text)
        if not json_match:
            raise ValueError("Valid JSON object not found in the response.")

        # Extract the JSON string
        json_string = json_match.group(0).strip()
        logging.info(f"Extracted JSON String:\n{json_string}")

        # Parse the JSON string
        parsed_json = json.loads(json_string)

        # Deduplicate nodes and edges
        nodes = {node["id"]: node for node in parsed_json.get("nodes", [])}.values()
        edges = {
            (edge["source"], edge["target"], edge["integration"], edge["context"]): edge
            for edge in parsed_json.get("edges", [])
        }.values()

        # Return the cleaned JSON object
        return {"nodes": list(nodes), "edges": list(edges)}

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {e}")
        raise ValueError("Error parsing JSON from response.")
    except Exception as e:
        logging.error(f"Error extracting JSON: {e}")
        raise
# In[12]: Bedrock method
import boto3
import os
import json
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def generate_json_with_bedrock(bedrock_client, inference_profile_arn, query, context):
    """
    Generate a JSON response using AWS Bedrock.

    Args:
        bedrock_client: AWS Bedrock client.
        inference_profile_arn (str): The ARN of the inference profile to use.
        query (str): User query.
        context (list): Context of interactions.

    Returns:
        dict: Generated JSON response.
    """
    try:
        # Create the prompt
        prompt = create_prompt(query, context)

        # Prepare request body
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 1500,
            "temperature": 0.1,
            "top_p": 0.9
        })

        # Invoke Bedrock
        response = bedrock_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
            contentType="application/json",
            accept="application/json",
        )

        # Parse the response body
        response_body = json.loads(response["body"].read())
        extracted_json = extract_json_from_response(response_body)

        return extracted_json

    except Exception as e:
        logger.error(f"Error generating JSON with Bedrock: {e}")
        raise
# %%
def generate_json_from_context(context):
    """
    Generate a JSON response with nodes and edges from the given context.

    Args:
        context (list): List of interactions containing source, target, integration, and context.

    Returns:
        dict: JSON object with deduplicated nodes and edges.
    """
    try:
        logging.info(f"Generating JSON from context: {context}")

        # Create nodes and edges
        nodes = {entry["Source"]: {"id": entry["Source"]} for entry in context}
        nodes.update({entry["Target"]: {"id": entry["Target"]} for entry in context})

        edges = [
            {
                "source": entry["Source"],
                "target": entry["Target"],
                "integration": entry["Integration"],
                "context": entry["Context"]
            }
            for entry in context
        ]

        # Deduplicate edges
        unique_edges = {
            (edge["source"], edge["target"], edge["integration"], edge["context"]): edge
            for edge in edges
        }.values()

        # Return final JSON
        return {"nodes": list(nodes.values()), "edges": list(unique_edges)}

    except Exception as e:
        logging.error(f"Error generating JSON from context: {e}")
        raise
