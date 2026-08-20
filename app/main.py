from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.groq_service import get_groq_service
from app.services.rag_service import get_rag_service


# =========================================
# FASTAPI APPLICATION
# =========================================

app = FastAPI(
    title="Alakart AI Chatbot",
    version="1.0.0"
)


# =========================================
# CORS
# React frontend → FastAPI backend
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# TEST GROQ REQUEST MODEL
# =========================================

class TestPromptRequest(BaseModel):
    prompt: str = Field(
        default="Hello",
        min_length=1
    )


# =========================================
# CHAT REQUEST MODEL
# =========================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1
    )


# =========================================
# CHAT RESPONSE MODEL
# =========================================

class ChatResponse(BaseModel):
    response: str


# =========================================
# TEST RAG REQUEST MODEL
# =========================================

class TestRagRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1
    )


# =========================================
# ROOT ENDPOINT
# =========================================

@app.get("/")
def read_root():
    return {
        "status": "Alakart AI Chatbot is running"
    }


# =========================================
# TEST GROQ - GET
# =========================================

@app.get("/test-groq")
def test_groq_get(
    prompt: str = Query(
        "Hello",
        description="Prompt to send to Groq"
    )
):
    """
    Developer endpoint to verify
    Groq LLM connectivity.
    """

    try:
        service = get_groq_service()

        response_text = service.generate_response(
            prompt
        )

        return {
            "status": "success",
            "model": service.model,
            "prompt": prompt,
            "response": response_text,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================
# TEST GROQ - POST
# =========================================

@app.post("/test-groq")
def test_groq_post(
    payload: TestPromptRequest
):
    """
    Developer endpoint to verify
    Groq LLM connectivity.
    """

    try:
        service = get_groq_service()

        response_text = service.generate_response(
            payload.prompt
        )

        return {
            "status": "success",
            "model": service.model,
            "prompt": payload.prompt,
            "response": response_text,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================
# MAIN CHAT ENDPOINT
#
# React
#   ↓
# FastAPI
#   ↓
# RAG SERVICE
#   ↓
# Intent + Retrieval + Groq
#   ↓
# Response
# =========================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
    payload: ChatRequest
):
    """
    Main chatbot endpoint.

    IMPORTANT:
    This endpoint does NOT decide the user's intent.

    The RAG service is responsible for:

    1. Understanding the query
    2. Detecting intent
    3. Choosing the correct retrieval strategy
    4. Retrieving relevant information
    5. Separating general medicine from
       Alakart product information
    6. Generating the final response
    """

    try:
        rag = get_rag_service()

        response_text = rag.handle_query(
            payload.message.strip()
        )

        return ChatResponse(
            response=response_text
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================
# DEVELOPER-ONLY RAG DEBUG ENDPOINT
#
# This endpoint is extremely important
# during our rebuild.
#
# It allows us to see:
#
# - user question
# - detected intent
# - retrieved chunks
# - metadata
# - final response
#
# The normal chatbot will NOT expose
# this information to the user.
# =========================================

@app.post("/test-rag")
def test_rag(
    payload: TestRagRequest
):
    """
    Developer-only endpoint for debugging
    the complete RAG pipeline.
    """

    try:
        rag = get_rag_service()

        result = rag.handle_query_debug(
            payload.message.strip()
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )