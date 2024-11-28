import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from query_engine import (
    generate_json_with_openai_few_shot,
    retrieve_context_from_vector_db,
    setup_vector_db,
    graph,
)
from fastapi.middleware.cors import CORSMiddleware

# uvicorn rest_api_server:app --reload

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

from pydantic import BaseModel

# Define the expected request body
class QueryRequest(BaseModel):
    query: str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Try to set up the vector database at server startup
vector_db_client = None
try:
    vector_db_client = setup_vector_db(graph)
    if vector_db_client:
        logging.info("Vector database setup successful!")
    else:
        logging.warning("Vector database setup failed. Continuing without it.")
except Exception as e:
    logging.error(f"Error initializing vector database: {e}")


# Middleware to log HTTP requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code} for {request.method} {request.url}")
    return response

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
        retrieved_context = retrieve_context_from_vector_db(vector_db_client, query)
        logging.info(f"Retrieved Context: {retrieved_context}")

        if not retrieved_context:
            logging.warning(f"No context found for query: {query}")
            return JSONResponse(
                content={"query": query, "nodes": [], "edges": []},
                status_code=200,
            )

        # Generate JSON response using OpenAI
        json_response = generate_json_with_openai_few_shot(retrieved_context, query)
        logging.info(f"Generated JSON Response: {json_response}")

        if json_response:
            return JSONResponse(content=json_response, status_code=200)
        else:
            logging.error("Failed to generate JSON response.")
            return JSONResponse(
                content={"query": query, "nodes": [], "edges": []},
                status_code=200,
            )

    except Exception as e:
        logging.exception(f"An error occurred while processing query: {query}")
        return JSONResponse(
            content={"error": str(e), "nodes": [], "edges": []},
            status_code=500,
        )

# Start the server using uvicorn:
# uvicorn rest_api_server:app --reload