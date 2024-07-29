import os
import logging
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import uvicorn
from pydantic import BaseModel
from typing import List
from datetime import date, datetime
import mysql.connector
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import traceback
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Pinecone initialization
pinecone_api_key = 'b807b048-2024-47bd-b4d5-c94e5f982ec0'
pinecone = Pinecone(api_key=pinecone_api_key)

index_name = "training-project-vectordb"
dimension = 384  # Update this with the dimensionality of your embeddings

# Directory for file uploads
UPLOADS_DIR = 'upload_files'
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize global stores
document_store = []
metadata_store = []

# Initialize the language model
openai_api_key = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert query results and user queries into natural language responses
def convert_to_natural_language(data: List[str], natural_language_text: str) -> str:
    try:
        if not data:
            return "No data present in the database for the given prompt. Please provide correct data."

        prompt_template = """
        You are a helpful assistant that converts database query results into natural language responses.
        Here is the natural language request:
        {natural_language_text}

        Here are the query results:
        {data}

        Instructions:
        - Review the query results and determine if they are relevant to the given prompt.
        - If the data is relevant, generate a coherent natural language response based on only the provided results.
        - If the data is not relevant or if no relevant data is found, respond with: "No information available, please provide a correct prompt."
        - Not add any additional information in response.
        
        Please provide a response in natural language based on these instructions.
        """

        # Define the prompt template
        prompt = PromptTemplate(input_variables=["natural_language_text", "data"], template=prompt_template)
        response_chain = LLMChain(prompt=prompt, llm=llm)

        # Format the data as a string for the prompt
        formatted_data = "\n\n".join([f"Source: {metadata}\nData:\n{data_item}" for data_item, metadata in zip(data, metadata_store)])

        result = response_chain.run(natural_language_text=natural_language_text, data=formatted_data)
        return result.strip()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

# Connect to MySQL
def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            port=5435,
            password='1234',
            database='task1'
        )
        return connection
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL: {err}")
        raise HTTPException(status_code=500, detail="Database connection error")

# Fetch data from MySQL
def fetch_all_tables_data():
    try:
        connection = connect_to_mysql()
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        all_data = []
        for table in tables:
            table_name = table[0]
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, connection)
            for index, row in df.iterrows():
                row_data = row.to_dict()
                all_data.append({"data": row_data, "source": f"MySQL table: {table_name}"})
                
        cursor.close()
        connection.close()
        return all_data
    except Exception as e:
        logger.error(f"Error fetching data from MySQL: {e}")
        raise HTTPException(status_code=500, detail="Error fetching data from database")

# Fetch data from files
def fetch_from_files(directory_path):
    try:
        data = []
        files = os.listdir(directory_path)
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    data.append({"data": content, "source": f"File: {file_name}"})
        return data
    except Exception as e:
        logger.error(f"Error reading files from {directory_path}: {e}")
        raise HTTPException(status_code=500, detail="Error reading files")

# Process and index data into chunks
def process_and_index_data(data):
    global document_store, metadata_store
    try:
        if not data:
            logger.warning("No data to index.")
            return

        chunk_size = 1000  # Define your chunk size
        for d in data:
            if isinstance(d["data"], dict):
                data_string = json.dumps(d["data"], default=json_serialize)
            else:
                data_string = d["data"]
            
            chunks = [data_string[i:i+chunk_size] for i in range(0, len(data_string), chunk_size)]
            for chunk in chunks:
                embeddings = model.encode([chunk])
                if embeddings.shape[0] == 0:
                    logger.warning("No embeddings to add to Pinecone.")
                    continue

                # Use upsert to add embeddings to Pinecone
                pinecone.Index(index_name).upsert(vectors=[{"id": str(len(document_store)), "values": embeddings[0].tolist()}])
                document_store.append(chunk)
                metadata_store.append(d["source"])

        logger.info("Data indexed successfully in Pinecone.")
    except Exception as e:
        logger.error(f"Error processing and indexing data: {e}")
        raise HTTPException(status_code=500, detail="Error processing and indexing data")

# Define custom JSON serializer for objects not serializable by default JSON encoder
def json_serialize(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def initialize_index():
    try:
        # Check if the index already exists
        if index_name not in pinecone.list_indexes().names():
            # Define the serverless specification
            spec = ServerlessSpec(region="us-east1-gcp", cloud="aws")

            # Create the index with the specified parameters
            pinecone.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=spec
            )
            print("Index created successfully.")
    except Exception as e:
        print(f"Error initializing Pinecone index: {e}")
        traceback.print_exc()  # Print stack trace for detailed error information
        raise HTTPException(status_code=500, detail="Error initializing Pinecone index")

# Define request and response models
class SearchQuery(BaseModel):
    query: str

class FileUploadResponse(BaseModel):
    filename: str

@app.post("/uploadfile/", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Process and index the uploaded file
        data = fetch_from_files(UPLOADS_DIR)
        process_and_index_data(data)

        return {"filename": file.filename}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Error uploading file")

@app.post("/deletefile/")
async def delete_file(filename: str):
    try:
        file_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

            # Remove the file data from Pinecone
            index = pinecone.Index(index_name)
            index.delete(ids=[filename])

            # Process and reindex remaining files
            remaining_files = os.listdir(UPLOADS_DIR)
            file_data = fetch_from_files(UPLOADS_DIR)
            process_and_index_data(file_data)

            return {"status": "success", "filename": filename}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail="Error deleting file")

def fetch_query_vector(query):
    try:
        query_vector = model.encode([query])
        return query_vector[0]
    except Exception as e:
        logger.error(f"Error encoding query: {e}")
        raise HTTPException(status_code=500, detail="Error encoding query")

logger = logging.getLogger(__name__)

@app.post("/query/")
async def query_data(search_query: SearchQuery):
    try:
        query_text = search_query.query
        query_vector = fetch_query_vector(query_text)

        # Ensure query_vector is in the correct format (flat list of floats)
        query_vector = query_vector.tolist()
        print("query vector is : ", query_vector)

        # Fetch relevant documents from Pinecone
        index = pinecone.Index(index_name)
        search_response = index.query(
            vector=query_vector,
            top_k=3,
            include_values=True
        )
        print("search response is : ", search_response)

        if 'matches' not in search_response or not search_response['matches']:
            logger.warning("No matches found.")
            return "No information available, please provide a correct prompt."

        matches = search_response['matches']
        
        # Ensure document_store is defined and accessible
        if not hasattr(document_store, '__len__'):
            logger.error("document_store is not defined or does not support length.")
            raise HTTPException(status_code=500, detail="Document store is not available.")
        
        # Validate indices and fetch documents
        data = []
        for match in matches:
            if 'id' in match:
                id = int(match['id'])
                if 0 <= id < len(document_store):
                    data.append(document_store[id])
                else:
                    logger.warning(f"Index {id} is out of range for document_store.")
            else:
                logger.warning(f"No 'id' field in match: {match}")

        if not data:
            return "No information available, please provide a correct prompt."

        return convert_to_natural_language(data, query_text)

    except Exception as e:
        logger.error(f"Error querying data: {e}")
        traceback.print_exc()  # Print stack trace for detailed error information
        raise HTTPException(status_code=500, detail="Error querying data")

@app.on_event("startup")
async def startup_event():
    initialize_index()
    file_data = fetch_from_files(UPLOADS_DIR)
    process_and_index_data(file_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
