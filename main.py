from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import ollama
import re
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import logging
from fastapi.responses import JSONResponse
from Text_to_Template_Conversion import convert_excel_to_text
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepSeek PDF Chat API",
    description="API for processing PDFs and answering questions using DeepSeek LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.mkdtemp()
MODEL_NAME = "deepseek-r1:1.5b"
BASE_DB_DIR = "./databases"

# Create base directory if it doesn't exist
os.makedirs(BASE_DB_DIR, exist_ok=True)


_dataframe_cache = {}


def get_db_path(database_name: str) -> str:
    """Get the path for a specific database"""
    return os.path.join(BASE_DB_DIR, database_name)


def process_pdf(pdf_path: str, database_name: str):
    """Process a PDF file and create a vector store from its contents."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(data)

        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        db_path = get_db_path(database_name)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )

        return text_splitter, vectorstore, vectorstore.as_retriever()
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise


def combine_docs(docs: list) -> str:
    """Combine multiple documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm(question: str, context: str) -> str:
    """Generate a response using the Ollama LLM."""
    try:
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': formatted_prompt}]
        )
        response_content = response["message"]["content"]
        return re.sub(
            r'<think>.*?</think>',
            '',
            response_content,
            flags=re.DOTALL
        ).strip()
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise


def rag_chain(question: str, database_name: str) -> str:
    """Run the retrieval-augmented generation chain."""
    try:
        db_path = get_db_path(database_name)
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=OllamaEmbeddings(model=MODEL_NAME)
        )
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(question)
        formatted_content = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_content)
    except Exception as e:
        logger.error(f"Error in RAG chain: {str(e)}")
        raise


class PDFUploadRequest(BaseModel):
    database_name: str


class QuestionRequest(BaseModel):
    question: str
    database_name: str


class ErrorResponse(BaseModel):
    error: str


@app.get("/databases/", response_model=List[str])
async def get_databases():
    """Get list of available databases"""
    try:
        if not os.path.exists(BASE_DB_DIR):
            return []
        databases = [d for d in os.listdir(BASE_DB_DIR)
                     if os.path.isdir(os.path.join(BASE_DB_DIR, d))]
        return databases
    except Exception as e:
        logger.error(f"Error getting databases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf/")
async def upload_pdf(
        file: UploadFile = File(...),
        database_name: str = Form(...)
) -> Dict[str, str]:
    """Upload and process a PDF file into a specific database."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        file_path = os.path.join(TEMP_DIR, file.filename)

        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error saving file")

        try:
            text_splitter, vectorstore, retriever = process_pdf(file_path, database_name)
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing PDF")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        return {"message": f"PDF processed successfully into database: {database_name}"}

    except Exception as e:
        logger.error(f"Unexpected error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/")
async def ask_question(question: QuestionRequest) -> Dict[str, str]:
    """Answer a question using a specific database."""
    try:
        result = rag_chain(question.question, question.database_name)
        return {"answer": result}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/database/{database_name}")
async def delete_database(database_name: str):
    """Delete a specific database."""
    try:
        db_path = get_db_path(database_name)
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            return {"message": f"Database {database_name} deleted successfully"}
        raise HTTPException(status_code=404, detail="Database not found")
    except Exception as e:
        logger.error(f"Error deleting database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred"}
    )


def process_text_files(output_dir: str, database_name: str):
    """Process text files generated from Excel and create a vector store."""
    try:
        text_data = []
        logger.info(f"Processing text files in directory: {output_dir}")
        
        for filename in os.listdir(output_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(output_dir, filename)
                logger.info(f"Reading file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as file:
                    text_data.append(file.read())

        if not text_data:
            logger.warning("No text files found in the directory.")
            return None, None, None

        logger.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.create_documents(text_data)

        logger.info("Generating embeddings...")
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        db_path = get_db_path(database_name)

        logger.info(f"Creating vector store at: {db_path}")
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=db_path
        )

        logger.info("Vector store created successfully.")
        return text_splitter, vectorstore, vectorstore.as_retriever()
    except Exception as e:
        logger.error(f"Error processing text files: {str(e)}")
        raise


def get_cached_dataframe(file_path):
    global _dataframe_cache

    try:
        file_mod_time = os.path.getmtime(file_path)
        if (
                file_path in _dataframe_cache
                and _dataframe_cache[file_path]["mod_time"] == file_mod_time
        ):
            return _dataframe_cache[file_path]["dataframe"]

        df = pd.read_excel(file_path, engine="openpyxl")
        df.columns = df.columns.str.lower().str.strip()

        if "pmid" in df.columns:
            df["pmid"] = df["pmid"].astype(str).str.strip().replace("-99", None)

        for col in df.columns:
            if col != "pmid":
                try:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = pd.to_numeric(df[col], errors="raise")
                    if df[col].dtype == "float64" and (df[col] % 1 == 0).all():
                        df[col] = df[col].astype("int64")
                    if df[col].dtype in ["int64", "float64"]:
                        df[col] = df[col].replace(-99, np.nan)
                except (ValueError, TypeError):
                    df[col] = df[col].replace("-99", None)

        _dataframe_cache[file_path] = {"dataframe": df, "mod_time": file_mod_time}
        return df

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise e


@app.post("/upload-excel/")
async def upload_excel(file: UploadFile = File(...), database_name: str = Form(...)):
    if not file.filename.endswith(('.xls', '.xlsx')):
        raise HTTPException(status_code=400, detail="File must be an Excel file")

    try:
        file_path = os.path.join(TEMP_DIR, file.filename)
        output_dir = os.path.join(TEMP_DIR, f"processed_{database_name}")

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        df = get_cached_dataframe(file_path)  # Обрабатываем файл перед преобразованием
        df.to_excel(file_path, index=False)  # Сохраняем обработанный файл

        convert_excel_to_text(df, output_dir)

        # Process text files and create vector store
        text_splitter, vectorstore, retriever = process_text_files(output_dir, database_name)

        return {"message": f"Excel processed successfully into database: {database_name}"}

    except Exception as e:
        logger.error(f"Unexpected error in upload_excel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import requests
    import asyncio
    from pathlib import Path


    # Test function
    def test_upload():
        url = "http://localhost:8000/upload-excel/"
        test_file = Path("D:\\000333\\deepseek_backend\\input\\TAF1_Upload_2025_02_03.xlsx")  # Change this to your test file path

        if not test_file.exists():
            print(f"Test file not found: {test_file}")
            return

        files = {
            'file': (
            'test.xlsx', open(test_file, 'rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        }
        data = {'database_name': 'test_db'}

        try:
            response = requests.post(url, files=files, data=data)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")


    # Run server and test
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000))

    asyncio.get_event_loop().run_until_complete(server.serve())
    test_upload()