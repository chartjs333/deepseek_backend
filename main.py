from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ollama
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import logging
from fastapi.responses import JSONResponse

# Configure logging
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory for PDF processing
TEMP_DIR = tempfile.mkdtemp()
MODEL_NAME = "deepseek-r1:1.5b"


def process_pdf(pdf_path: str):
    """
    Process a PDF file and create a vector store from its contents.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        tuple: Contains text_splitter, vectorstore, and retriever objects
    """
    try:
        loader = PyMuPDFLoader(pdf_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(data)

        embeddings = OllamaEmbeddings(model=MODEL_NAME)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        return text_splitter, vectorstore, vectorstore.as_retriever()
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise


def combine_docs(docs: list) -> str:
    """Combine multiple documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm(question: str, context: str) -> str:
    """
    Generate a response using the Ollama LLM.

    Args:
        question (str): The user's question
        context (str): The context from retrieved documents

    Returns:
        str: The generated response
    """
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


def rag_chain(question: str, text_splitter, vectorstore, retriever) -> str:
    """
    Run the retrieval-augmented generation chain.

    Args:
        question (str): The user's question
        text_splitter: The text splitter object
        vectorstore: The vector store object
        retriever: The retriever object

    Returns:
        str: The generated response
    """
    try:
        retrieved_docs = retriever.invoke(question)
        formatted_content = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_content)
    except Exception as e:
        logger.error(f"Error in RAG chain: {str(e)}")
        raise


class QuestionRequest(BaseModel):
    question: str


class ErrorResponse(BaseModel):
    error: str


@app.post("/upload-pdf/",
          response_model=Dict[str, str],
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload and process a PDF file.

    Args:
        file (UploadFile): The PDF file to process

    Returns:
        dict: A message indicating success or error
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        file_path = os.path.join(TEMP_DIR, file.filename)

        # Save uploaded file
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error saving file")

        # Process the PDF
        try:
            text_splitter, vectorstore, retriever = process_pdf(file_path)
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing PDF")
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

        return {"message": "PDF processed successfully"}

    except Exception as e:
        logger.error(f"Unexpected error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/",
          response_model=Dict[str, str],
          responses={
              500: {"model": ErrorResponse}
          })
async def ask_question(question: QuestionRequest) -> Dict[str, str]:
    """
    Answer a question about the processed PDF.

    Args:
        question (QuestionRequest): The question to answer

    Returns:
        dict: The answer to the question
    """
    try:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=OllamaEmbeddings(model=MODEL_NAME)
        )
        retriever = vectorstore.as_retriever()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        result = rag_chain(question.question, text_splitter, vectorstore, retriever)
        return {"answer": result}

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)