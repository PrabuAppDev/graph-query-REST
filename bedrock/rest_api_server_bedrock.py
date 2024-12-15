import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from query_engine_bedrock import setup_vector_db, query_bedrock
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

from pydantic import BaseModel
class QueryRequest(BaseModel):
    query: str

vector_db_client = None

def load_csv_and_setup_db():
    global vector_db_client
    try:
        csv_file = "sample_integration_data.csv"
        data = pd.read_csv(csv_file)
        print("CSV Data Loaded Successfully:")
        print(data.head())  # Show first few rows of the dataframe        
        edges = [row.to_dict() for index, row in data.iterrows()]
        vector_db_client = setup_vector_db(edges, drop_existing=True)
        logging.info("Vector database setup complete.")
    except Exception as e:
        logging.error(f"Error setting up vector database: {e}")

load_csv_and_setup_db()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code} for {request.method} {request.url}")
    return response

@app.post("/query")
async def query_interactions(request: QueryRequest):
    query = request.query
    logging.info(f"Received query: {query}")
    if vector_db_client is None:
        logging.warning("Vector database is unavailable.")
        return JSONResponse(content={"query": query, "nodes": [], "edges": []}, status_code=503)
    try:
        response_data = query_bedrock(vector_db_client, query)
        return JSONResponse(content=response_data, status_code=200)
    except Exception as e:
        logging.exception(f"An error occurred while processing query: {query}")
        return JSONResponse(content={"error": str(e), "nodes": [], "edges": []}, status_code=500)

# To run the server, use: uvicorn rest_api_server_bedrock:app --reload