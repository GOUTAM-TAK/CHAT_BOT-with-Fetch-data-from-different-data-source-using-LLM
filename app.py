import os
import json
import faiss
import logging
import pandas as pd
import numpy as np
import mysql.connector
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import uvicorn
from pydantic import BaseModel
from typing import List
from datetime import date, datetime

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

# Configuration for MySQL
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PORT = 5435
MYSQL_PASSWORD = '1234'
MYSQL_DATABASE = 'task1'

# Directory for file uploads and FAISS index
UPLOADS_DIR = 'upload_files'
INDEX_DIR = 'faiss_index'
INDEX_FILE = os.path.join(INDEX_DIR, 'faiss_index.bin')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize the FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Stores for document data and metadata
document_store = []
metadata_store = []

# Initialize the language model
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# Custom JSON serializer for objects not serializable by default JSON encoder
def json_serialize(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# Token counting utility
def count_tokens(text: str) -> int:
    # Placeholder for token counting logic
    return len(text.split())

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
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port=MYSQL_PORT,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
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
                    logger.warning("No embeddings to add to FAISS.")
                    continue

                index.add(embeddings)
                document_store.append(chunk)
                metadata_store.append(d["source"])

        # Save the index to disk
        faiss.write_index(index, INDEX_FILE)
        logger.info("Data indexed successfully in FAISS index and saved to disk.")
    except Exception as e:
        logger.error(f"Error processing and indexing data: {e}")
        raise HTTPException(status_code=500, detail="Error processing and indexing data")

# Load the FAISS index from disk
def load_faiss_index():
    global index, document_store, metadata_store
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        logger.info("FAISS index loaded from disk.")
    else:
        logger.info("FAISS index not found on disk. Creating a new index.")
        mysql_data = fetch_all_tables_data()
        process_and_index_data(mysql_data)
        file_data = fetch_from_files(UPLOADS_DIR)
        process_and_index_data(file_data)

# Initial data fetching and indexing
load_faiss_index()

class QueryRequest(BaseModel):
    prompt: str

@app.post("/query")
async def query(request: QueryRequest):
    prompt = request.prompt
    try:
        # Search in FAISS index
        query_embedding = model.encode([prompt])
        distances, indices = index.search(query_embedding, k=10)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(document_store):
                result_data = document_store[idx]
                result_metadata = metadata_store[idx]
                results.append({
                    "distance": dist,
                    "data": result_data,
                    "metadata": result_metadata
                })

        if not results:
            return {"response": "No data found for the given query."}

        # Convert results to natural language response
        relevant_data = [res["data"] for res in results]
        response = convert_to_natural_language(relevant_data, prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")

@app.get("/list-files/")
async def list_files():
    try:
        files = os.listdir(UPLOADS_DIR)
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail="Error listing files")

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        # Process and index the uploaded file data
        file_data = fetch_from_files(UPLOADS_DIR)
        process_and_index_data(file_data)

        return {"info": f"File '{file.filename}' uploaded and indexed successfully"}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Error uploading file")

@app.delete("/delete-file/{filename}")
async def delete_file(filename: str):
    try:
        file_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

            # Clear existing index and metadata stores
            global index, document_store, metadata_store
            index = faiss.IndexFlatL2(dimension)
            document_store.clear()
            metadata_store.clear()

            # Rebuild the index from remaining files
            file_data = fetch_from_files(UPLOADS_DIR)
            process_and_index_data(file_data)

            return {"info": f"File '{filename}' deleted and FAISS index updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail="Error deleting file")

@app.post("/update-index/")
async def update_index():
    try:
        global index, document_store, metadata_store
        index = faiss.IndexFlatL2(dimension)
        document_store.clear()
        metadata_store.clear()

        # Fetch and index data from MySQL and files
        mysql_data = fetch_all_tables_data()
        process_and_index_data(mysql_data)
        file_data = fetch_from_files(UPLOADS_DIR)
        process_and_index_data(file_data)

        return {"info": "FAISS index updated successfully with data from MySQL and files"}
    except Exception as e:
        logger.error(f"Error updating FAISS index: {e}")
        raise HTTPException(status_code=500, detail="Error updating FAISS index")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
