import logging
import json
import os
import boto3
from botocore.exceptions import ClientError
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
vectorizer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")  # Default to "us-east-2" if not set
AWS_PROFILE = "AWS_PROFILE_CLI"  # Default to "default" profile if not set
AWS_BEDROCK_AGENT_ID = os.getenv("AWS_BEDROCK_AGENT_ID")
AWS_BEDROCK_AGENT_ALIAS_ID = os.getenv("AWS_BEDROCK_AGENT_ALIAS_ID")

# Validate required environment variables
assert AWS_BEDROCK_AGENT_ID, "AWS_BEDROCK_AGENT_ID is not set"
assert AWS_BEDROCK_AGENT_ALIAS_ID, "AWS_BEDROCK_AGENT_ALIAS_ID is not set"

# Initialize Bedrock Agent Runtime client using AWS profile
try:
    session = boto3.Session(profile_name=AWS_PROFILE)
    logging.info(f"Initialized AWS session using profile: {AWS_PROFILE}")
    bedrock_agent_runtime_client = session.client(
        service_name="bedrock-agent-runtime",
        region_name=AWS_REGION
    )
    logging.info("AWS Bedrock Agent Runtime client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Bedrock Agent Runtime client: {e}")
    raise

def setup_vector_db(edges, drop_existing=True):
    client = QdrantClient(path=":memory:")
    print("Qdrant Client initialized successfully.")
    if drop_existing and client.collection_exists("systems"):
        print("Deleting existing collection...")
        client.delete_collection(collection_name="systems")
        print("Existing collection deleted.")
    else:
        print("No existing collection found.")
    client.create_collection(collection_name="systems", vectors_config=VectorParams(size=384, distance="Cosine"))
    print("Collection 'systems' created successfully.")
    populate_vector_db(client, edges)
    return client

def populate_vector_db(client, edges):
    points = []
    try:
        print(f"Number of edges to process: {len(edges)}")

        for idx, edge in enumerate(edges):
            # Normalize keys
            normalized_edge = {
                "consumer": edge.get("Consumer"),
                "producer": edge.get("Producer"),
                "integration": edge.get("Integration Type")
            }
            print(f"Processing edge {idx}: {normalized_edge}")

            # Validate the normalized edge
            if not all(key in normalized_edge and normalized_edge[key] for key in ['consumer', 'producer', 'integration']):
                raise ValueError(f"Edge {idx} is missing required keys: {normalized_edge}")

            # Construct the description for vectorization
            description = f"{normalized_edge['consumer']} interacts with {normalized_edge['producer']} via {normalized_edge['integration']}"
            print(f"Edge description for vectorization: {description}")

            # Generate vector for the description
            vector = vectorizer.encode(description).tolist()
            print(f"Generated vector (first 5 dimensions) for edge {idx}: {vector[:5]}")

            # Validate vector size
            expected_vector_size = 384
            if len(vector) != expected_vector_size:
                raise ValueError(f"Vector size mismatch for edge {idx}: expected {expected_vector_size}, got {len(vector)}")

            # Create PointStruct and add to the list
            point = PointStruct(
                id=idx,
                vector=vector,
                payload=normalized_edge
            )
            points.append(point)

        print(f"Number of points to upsert: {len(points)}")
        client.upsert(collection_name="systems", points=points)
        print("Upsert complete. Points have been added to the Qdrant collection.")

    except Exception as e:
        print(f"An error occurred in populate_vector_db: {e}")
        raise

def retrieve_context(client, query):
    vector = vectorizer.encode(query).tolist()
    results = client.search(collection_name="systems", query_vector=vector, limit=5)
    return [result.payload for result in results]

def invoke_agent_with_runtime(query, context):
    try:
        formatted_context = "\n".join([
            f"Consumer: {item['consumer']}, Producer: {item['producer']}, Integration: {item['integration']}"
            for item in context
        ])
        input_text = f"Query: {query}\nContext:\n{formatted_context}"
        logging.info("Input Text Sent to Agent:")
        logging.info(input_text)
        response = bedrock_agent_runtime_client.invoke_agent(
            agentId=AWS_BEDROCK_AGENT_ID,
            agentAliasId=AWS_BEDROCK_AGENT_ALIAS_ID,
            sessionId="test-session",
            inputText=input_text
        )
        logging.info("Full Raw Response from Bedrock Agent:")
        logging.info(response)
        completion = ""
        for event in response.get("completion"):
            chunk = event["chunk"]
            completion += chunk["bytes"].decode()
        return completion
    except ClientError as e:
        logging.error(f"Couldn't invoke agent. {e}")
        raise

def query_bedrock(client, query):
    context = retrieve_context(client, query)
    if not context:
        return None
    response = invoke_agent_with_runtime(query, context)
    return json.loads(response)