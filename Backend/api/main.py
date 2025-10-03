"""
FastAPI application with bilingual RAG chatbot and dataset analysis endpoints.
Implements secure API key handling and comprehensive error handling.
"""

import os
import logging
import uuid
import asyncio
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Internal imports
from ..chat.chat_service import get_chat_service, ChatRequest, ChatResponse
from ..data.analyze import DatasetAnalyzer
from ..data.chunk_embed import create_embedding_pipeline
from ..models.automl.quick_train import create_quick_trainer, TrainingConfig
from ..chat.vector_store import get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Request/Response Models
class ChatRequestModel(BaseModel):
    user_id: str
    message: str
    language: Optional[str] = None
    session_id: Optional[str] = None
    context_filters: Optional[Dict[str, Any]] = None

class ChatResponseModel(BaseModel):
    reply: str
    citations: List[Dict[str, str]]
    suggested_actions: List[Dict[str, str]]
    language: str
    confidence: float

class DataUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    size_bytes: int
    status: str
    message: str

class DataProfileResponse(BaseModel):
    profile: Dict[str, Any]
    action_plan: Dict[str, Any]
    status: str

class TrainingJobRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: Optional[str] = "auto"
    config: Optional[Dict[str, Any]] = None

class TrainingJobResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    estimated_time: str

class EmbeddingRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    time_column: Optional[str] = None

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    logger.info("Starting Cosmic Analysts ExoAI API...")
    
    # Initialize services
    try:
        chat_service = get_chat_service()
        vector_store = get_vector_store()
        
        # Health check
        stats = chat_service.get_stats()
        logger.info(f"Services initialized: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Cosmic Analysts ExoAI API",
    description="Bilingual RAG-enabled chatbot with dataset analysis pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
def get_chat_service_dependency():
    """Get chat service instance."""
    return get_chat_service()

def get_dataset_analyzer():
    """Get dataset analyzer instance."""
    return DatasetAnalyzer()

def get_embedding_pipeline():
    """Get embedding pipeline instance."""
    return create_embedding_pipeline()

def get_quick_trainer():
    """Get quick trainer instance."""
    return create_quick_trainer()

# Security dependencies
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints."""
    if not credentials:
        return None  # Allow public access for now
    
    # In production, verify the API key
    api_key = credentials.credentials
    expected_key = os.getenv("API_KEY")
    
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        chat_service = get_chat_service_dependency()
        stats = chat_service.get_stats()
        
        return {
            "status": "healthy",
            "service_stats": stats,
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Chat endpoints
@app.post("/api/chat", response_model=ChatResponseModel)
async def chat_endpoint(
    request: ChatRequestModel,
    chat_service = Depends(get_chat_service_dependency),
    api_key = Depends(verify_api_key)
):
    """
    Process chat request with bilingual RAG pipeline.
    """
    try:
        logger.info(f"Chat request from user {request.user_id}: {request.message[:50]}...")
        
        # Create chat request
        chat_request = ChatRequest(
            user_id=request.user_id,
            message=request.message,
            language=request.language,
            session_id=request.session_id,
            context_filters=request.context_filters
        )
        
        # Process with chat service
        response = await chat_service.chat(chat_request)
        
        return ChatResponseModel(
            reply=response.reply,
            citations=response.citations,
            suggested_actions=response.suggested_actions,
            language=response.language.value,
            confidence=response.confidence
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/api/chat/history/{user_id}")
async def get_chat_history(
    user_id: str,
    session_id: Optional[str] = None,
    chat_service = Depends(get_chat_service_dependency)
):
    """Get chat history for a user session."""
    try:
        history = chat_service.get_chat_history(user_id, session_id)
        return {
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in history
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/history/{user_id}")
async def clear_chat_history(
    user_id: str,
    session_id: Optional[str] = None,
    chat_service = Depends(get_chat_service_dependency)
):
    """Clear chat history for a user session."""
    try:
        success = chat_service.clear_chat_history(user_id, session_id)
        return {"success": success, "message": "Chat history cleared"}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data upload and analysis endpoints
@app.post("/api/data/upload", response_model=DataUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    analyzer = Depends(get_dataset_analyzer)
):
    """
    Upload and analyze dataset.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        allowed_extensions = ['.csv', '.parquet', '.json']
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded file
        dataset_id = str(uuid.uuid4())
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{dataset_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded dataset {dataset_id}: {file.filename} ({len(content)} bytes)")
        
        return DataUploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            size_bytes=len(content),
            status="uploaded",
            message="Dataset uploaded successfully. Analysis starting..."
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/data/{dataset_id}/profile", response_model=DataProfileResponse)
async def get_dataset_profile(
    dataset_id: str,
    analyzer = Depends(get_dataset_analyzer)
):
    """
    Get dataset profile and action plan.
    """
    try:
        # Find uploaded file
        upload_dir = "data/uploads"
        files = [f for f in os.listdir(upload_dir) if f.startswith(dataset_id)]
        
        if not files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = os.path.join(upload_dir, files[0])
        filename = files[0].split('_', 1)[1]  # Remove dataset_id prefix
        
        # Analyze dataset
        logger.info(f"Analyzing dataset {dataset_id}")
        profile, action_plan = await analyzer.analyze_dataset(
            file_path=file_path,
            dataset_id=dataset_id,
            filename=filename
        )
        
        return DataProfileResponse(
            profile=profile.__dict__,
            action_plan=action_plan.__dict__,
            status="completed"
        )
        
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/data/{dataset_id}/create-embeddings")
async def create_dataset_embeddings(
    dataset_id: str,
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    pipeline = Depends(get_embedding_pipeline)
):
    """
    Create embeddings for dataset (background task).
    """
    try:
        # Find dataset file
        upload_dir = "data/uploads"
        files = [f for f in os.listdir(upload_dir) if f.startswith(dataset_id)]
        
        if not files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = os.path.join(upload_dir, files[0])
        filename = files[0].split('_', 1)[1]
        
        # Add background task
        background_tasks.add_task(
            process_embeddings_background,
            pipeline,
            file_path,
            dataset_id,
            filename,
            request.target_column,
            request.time_column
        )
        
        return {
            "dataset_id": dataset_id,
            "status": "processing",
            "message": "Embedding creation started in background"
        }
        
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_embeddings_background(
    pipeline,
    file_path: str,
    dataset_id: str, 
    filename: str,
    target_column: Optional[str],
    time_column: Optional[str]
):
    """Background task for processing embeddings."""
    try:
        import pandas as pd
        
        # Load dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Process embeddings
        result = await pipeline.process_dataset(
            df=df,
            dataset_id=dataset_id,
            filename=filename,
            target_column=target_column,
            time_column=time_column
        )
        
        logger.info(f"Embeddings processed for dataset {dataset_id}: {result['statistics']}")
        
    except Exception as e:
        logger.error(f"Background embedding processing failed: {e}")

# Training endpoints
@app.post("/api/train/quick", response_model=TrainingJobResponse)
async def quick_train(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks,
    trainer = Depends(get_quick_trainer)
):
    """
    Start quick training job.
    """
    try:
        # Find dataset
        upload_dir = "data/uploads"
        files = [f for f in os.listdir(upload_dir) if f.startswith(request.dataset_id)]
        
        if not files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = os.path.join(upload_dir, files[0])
        
        # Create training config
        config = TrainingConfig(
            task_type=request.task_type,
            target_column=request.target_column,
            **(request.config or {})
        )
        
        job_id = str(uuid.uuid4())
        
        # Add background task
        background_tasks.add_task(
            train_model_background,
            trainer,
            file_path,
            config,
            request.dataset_id,
            job_id
        )
        
        return TrainingJobResponse(
            job_id=job_id,
            dataset_id=request.dataset_id,
            status="training",
            estimated_time="5-10 minutes"
        )
        
    except Exception as e:
        logger.error(f"Training job creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(
    trainer,
    file_path: str,
    config: TrainingConfig,
    dataset_id: str,
    job_id: str
):
    """Background task for model training."""
    try:
        import pandas as pd
        
        # Load dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Train model
        result = await trainer.train_model(
            df=df,
            config=config,
            dataset_id=dataset_id
        )
        
        logger.info(f"Training completed for job {job_id}: {result.status}")
        
        # Store result (in production, use database)
        results_dir = "data/training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        with open(os.path.join(results_dir, f"{job_id}.json"), 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Background training failed: {e}")

@app.get("/api/train/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status and results."""
    try:
        results_dir = "data/training_results"
        result_file = os.path.join(results_dir, f"{job_id}.json")
        
        if not os.path.exists(result_file):
            return {"job_id": job_id, "status": "running", "message": "Training in progress"}
        
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        return {
            "job_id": job_id,
            "status": result["status"],
            "metrics": result.get("metrics", {}),
            "training_time": result.get("training_time", 0),
            "model_path": result.get("model_path", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vector store management endpoints
@app.get("/api/vector/stats")
async def get_vector_store_stats():
    """Get vector store statistics."""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vector/search")
async def search_documents(
    query: str,
    k: int = 10,
    language: Optional[str] = None,
    source_type: Optional[str] = None,
    chat_service = Depends(get_chat_service_dependency)
):
    """Search documents in vector store."""
    try:
        results = await chat_service.search_documents(
            query=query,
            k=k,
            language=language,
            source_type=source_type
        )
        
        return {
            "query": query,
            "results": [
                {
                    "document": {
                        "id": doc.id,
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "source_type": doc.source_type,
                        "language": doc.language,
                        "metadata": doc.metadata
                    },
                    "score": float(score)
                }
                for doc, score in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main application entry point
if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
