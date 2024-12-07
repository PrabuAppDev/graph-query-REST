import logging
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from query_engine_simple import setup_vector_db, retrieve_context
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define the expected request body
from pydantic import BaseModel
class QueryRequest(BaseModel):
    query: str

# Initialize vector db client globally
vector_db_client = None

def load_csv_and_setup_db():
    """
    Load CSV and setup the vector database at server startup.
    """
    global vector_db_client
    try:
        # Read CSV file and convert to list of dicts
        csv_file = "sample_integration_data.csv"
        data = pd.read_csv(csv_file)

        edges = []
        for _, row in data.iterrows():
            edges.append({
                "consumer": row["Consumer"],
                "producer": row["Producer"],
                "integration": row["Integration Type"],
            })

        # Setup the vector database with the edges
        vector_db_client = setup_vector_db(edges, drop_existing=True)
        logging.info("Vector database setup complete.")

    except Exception as e:
        logging.error(f"Error setting up vector database: {e}")

# Load the CSV and setup the vector database on startup
load_csv_and_setup_db()

# Middleware to log HTTP requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code} for {request.method} {request.url}")
    return response

# Function to parse the context and prepare it for D3 visualization
def parse_context_for_d3(retrieved_context):
    """
    Parse the retrieved context to extract nodes and edges for D3.js.

    Args:
        retrieved_context (list): List of dictionaries containing system interactions.

    Returns:
        dict: A dictionary with 'nodes' and 'edges' ready for D3.js.
    """
    nodes = {}
    edges = []

    # Extract nodes and edges from the retrieved context
    for entry in retrieved_context:
        consumer = entry.get('consumer')
        producer = entry.get('producer')
        integration = entry.get('integration')

        # Add consumer and producer to the nodes dictionary (avoiding duplicates)
        if consumer not in nodes:
            nodes[consumer] = {"id": consumer}
        if producer not in nodes:
            nodes[producer] = {"id": producer}

        # Create the edge between consumer and producer
        edge = {
            "consumer": consumer,
            "producer": producer,
            "integration": integration
        }
        edges.append(edge)

    return {"nodes": list(nodes.values()), "edges": edges}

@app.post("/query")
async def query_interactions(request: QueryRequest):
    """
    Endpoint to retrieve system interactions for a given query.
    """
    query = request.query
    logging.info(f"Received query: {query}")

    if vector_db_client is None:
        logging.warning("Vector database is unavailable.")
        return JSONResponse(
            content={"query": query, "nodes": [], "edges": []},
            status_code=503,
        )

    try:
        # Retrieve context from the vector database
        retrieved_context, edges = retrieve_context(vector_db_client, query)
        logging.info(f"Retrieved Context: {retrieved_context}")
        logging.info(f"Retrieved Edges: {edges}")

        if not retrieved_context:
            logging.warning(f"No context found for query: {query}")
            return JSONResponse(
                content={"query": query, "nodes": [], "edges": []},
                status_code=200,
            )

        # Parse context for D3.js
        parsed_data = parse_context_for_d3(retrieved_context)
        logging.info(f"Parsed Data: {parsed_data}")

        # Pretty-print the JSON response in the logger
        json_response = {
            "query": query,
            "nodes": parsed_data['nodes'],
            "edges": parsed_data['edges']
        }

        logging.info(f"Generated JSON Response: {json.dumps(json_response, indent=2)}")

        return JSONResponse(content=json_response, status_code=200)

    except Exception as e:
        logging.exception(f"An error occurred while processing query: {query}")
        return JSONResponse(
            content={"error": str(e), "nodes": [], "edges": []},
            status_code=500,
        )

# To run the server, use the following command:
# uvicorn rest_api_server_simple:app --reload