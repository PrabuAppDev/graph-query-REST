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
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_BEDROCK_AGENT_ID = os.getenv("AWS_BEDROCK_AGENT_ID")
AWS_BEDROCK_AGENT_ALIAS_ID = os.getenv("AWS_BEDROCK_AGENT_ALIAS_ID")

# Validate environment variables
assert AWS_BEDROCK_AGENT_ID, "AWS_BEDROCK_AGENT_ID is not set"
assert AWS_BEDROCK_AGENT_ALIAS_ID, "AWS_BEDROCK_AGENT_ALIAS_ID is not set"

# Initialize Bedrock Agent Runtime client
bedrock_agent_runtime_client = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

logging.info("Bedrock Agent Runtime client initialized successfully!")

def setup_vector_db(edges, drop_existing=True):
    client = QdrantClient(path=":memory:")
    if drop_existing and client.collection_exists("systems"):
        client.delete_collection(collection_name="systems")
    client.create_collection(collection_name="systems", vectors_config=VectorParams(size=384, distance="Cosine"))
    populate_vector_db(client, edges)
    return client

def populate_vector_db(client, edges):
    points = []
    for idx, edge in enumerate(edges):
        description = f"{edge['consumer']} interacts with {edge['producer']} via {edge['integration']}"
        vector = vectorizer.encode(description).tolist()
        points.append(PointStruct(id=idx, vector=vector, payload=edge))
    client.upsert(collection_name="systems", points=points)

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