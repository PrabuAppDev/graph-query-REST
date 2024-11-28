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

@app.post("/query")
async def query_interactions(request: QueryRequest):
    query = request.query
    # The rest of your logic

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
@app.post("/query")
async def query_interactions(request: QueryRequest):
    query = request.query

    if vector_db_client is None:
        logging.warning("Vector database is unavailable.")
        return JSONResponse(
            content={"error": "Vector database not available.", "nodes": [], "edges": []},
            status_code=503,
        )

    try:
        # Retrieve context from vector database
        retrieved_context = retrieve_context_from_vector_db(vector_db_client, query)
        if not retrieved_context:
            logging.warning(f"No context found for query: {query}")
            return JSONResponse(
                content={"query": query, "nodes": [], "edges": []},
                status_code=200,
            )

        # Generate JSON response using OpenAI API
        json_response = generate_json_with_openai_few_shot(retrieved_context, query)
        if json_response:
            logging.info(f"Successfully generated JSON response for query: {query}")
            return JSONResponse(content=json_response, status_code=200)
        else:
            logging.error(f"Failed to generate JSON response for query: {query}")
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