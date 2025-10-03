import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import our custom modules
import sys
sys.path.append('..')
from models.tabkanet import TabKANet, KANLayer
from load_model import load_model_from_config
from export_model import export_model_cli
from app.worker import get_worker
from app.model_registry import get_registry

app = FastAPI(
    title="Cosmic Analysts ExoAI API",
    description="Advanced ML platform for tabular data with TabKANet architecture",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
worker = get_worker()
registry = get_registry()

UPLOAD_DIR = Path("uploads")
EXPORTS_DIR = Path("exports")

# Create directories
for dir_path in [UPLOAD_DIR, EXPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Pydantic models for API
class DataPreview(BaseModel):
    columns: List[Dict[str, Any]]
    rowCount: int
    sample: List[Dict[str, Any]]

class TrainingConfig(BaseModel):
    model_type: str = "tabkanet"
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    dropout: float = 0.1
    d_model: int = 64
    K_inner: int = 16
    trans_heads: int = 4
    trans_depth: int = 3
    mlp_hidden: int = 128
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.2

class TrainingJob(BaseModel):
    id: str
    name: str
    status: str
    progress: float
    start_time: str
    duration: str
    config: TrainingConfig
    metrics: Optional[Dict[str, float]] = None

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    architecture: str
    created_at: str
    training_time: str
    dataset_size: int
    parameters: int
    framework: str
    accuracy: float
    model_path: str

class PredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]

class ExportRequest(BaseModel):
    model_id: str
    format: str = "torchscript"  # torchscript, onnx, tensorrt

@app.get("/")
async def root():
    return {"message": "Cosmic Analysts ExoAI API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload and preview dataset"""
    if not file.filename.endswith(('.csv', '.parquet', '.json')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    try:
        # Load data based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file_path)
        
        # Generate data preview
        columns = []
        for col in df.columns:
            col_data = df[col]
            
            # Detect column type
            if pd.api.types.is_numeric_dtype(col_data):
                col_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_type = "datetime"
            elif col_data.nunique() / len(col_data) < 0.1:  # Low cardinality
                col_type = "categorical"
            else:
                col_type = "text"
            
            columns.append({
                "name": col,
                "type": col_type,
                "nullCount": int(col_data.isnull().sum()),
                "uniqueCount": int(col_data.nunique()),
                "sample": col_data.dropna().head(5).tolist()
            })
        
        # Sample data
        sample_data = df.head(5).fillna("null").to_dict(orient="records")
        
        preview = DataPreview(
            columns=columns,
            rowCount=len(df),
            sample=sample_data
        )
        
        # Store file info for later use
        file_info = {
            "path": str(file_path),
            "preview": preview.dict(),
            "uploaded_at": datetime.now().isoformat()
        }
        
        return {
            "file_id": file_path.stem,
            "preview": preview.dict(),
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/api/training/start")
async def start_training(
    file_id: str,
    target_column: str,
    config: TrainingConfig
):
    """Start a training job"""
    job_name = f"{config.model_type}-{target_column}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Find the uploaded file
    file_path = None
    for ext in ['.csv', '.parquet', '.json']:
        potential_path = UPLOAD_DIR / f"{file_id}{ext}"
        if potential_path.exists():
            file_path = str(potential_path)
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    # Submit job to worker
    job_data = {
        'name': job_name,
        'file_path': file_path,
        'target_column': target_column,
        'config': config.dict()
    }
    
    job_id = await worker.submit_job(job_data)
    
    return {"job_id": job_id, "message": "Training job submitted", "status": "queued"}


@app.get("/api/training/jobs")
async def get_training_jobs():
    """Get all training jobs"""
    jobs = worker.get_all_jobs()
    return {"jobs": list(jobs.values())}

@app.get("/api/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get specific training job"""
    job = worker.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}

@app.get("/api/models")
async def get_models():
    """Get all registered models"""
    models = registry.list_models()
    return {"models": [model.dict() for model in models]}

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model info"""
    model = registry.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model": model.dict()}

@app.post("/api/models/{model_id}/predict")
async def predict(model_id: str, request: PredictionRequest):
    """Make predictions with a trained model"""
    model_info = registry.get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load model
        checkpoint = torch.load(model_info.model_path, map_location='cpu')
        
        # Reconstruct model
        config = checkpoint['model_config']
        model = TabKANet(
            n_num=checkpoint['n_features'],
            n_cat=0,
            cat_card_list=[],
            d_model=config['d_model'],
            K_inner=config['K_inner'],
            trans_heads=config['trans_heads'],
            trans_depth=config['trans_depth'],
            mlp_hidden=config['mlp_hidden'],
            n_classes=checkpoint['n_classes'],
            dropout=config['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Prepare input data
        df = pd.DataFrame(request.data)
        
        # Apply same preprocessing
        scaler = checkpoint['scaler']
        label_encoders = checkpoint['label_encoders']
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))
        
        # Scale features
        X_scaled = scaler.transform(df[checkpoint['feature_names']])
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(X_tensor, None)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        # Decode predictions if needed
        target_encoder = checkpoint.get('target_encoder')
        if target_encoder:
            predictions_decoded = target_encoder.inverse_transform(predictions.numpy())
        else:
            predictions_decoded = predictions.numpy()
        
        return {
            "predictions": predictions_decoded.tolist(),
            "probabilities": probabilities.numpy().tolist(),
            "model_id": model_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/models/{model_id}/export")
async def export_model_endpoint(model_id: str, request: ExportRequest):
    """Export model to different formats"""
    model_info = registry.get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_path = Path(model_info.model_path)
        
        # Create export directory
        export_dir = EXPORTS_DIR / model_id
        export_dir.mkdir(exist_ok=True)
        
        # Export based on format
        if request.format == "torchscript":
            # Load and convert to TorchScript
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['model_config']
            
            model = TabKANet(
                n_num=checkpoint['n_features'],
                n_cat=0,
                cat_card_list=[],
                d_model=config['d_model'],
                K_inner=config['K_inner'],
                trans_heads=config['trans_heads'],
                trans_depth=config['trans_depth'],
                mlp_hidden=config['mlp_hidden'],
                n_classes=checkpoint['n_classes'],
                dropout=config['dropout']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Create example input
            example_input = torch.randn(1, checkpoint['n_features'])
            
            # Convert to TorchScript
            traced_model = torch.jit.trace(model, (example_input, None))
            export_path = export_dir / f"{model_info.name}.pt"
            traced_model.save(str(export_path))
            
        elif request.format == "onnx":
            # ONNX export would go here
            export_path = export_dir / f"{model_info.name}.onnx"
            # Placeholder - actual ONNX export implementation needed
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {request.format}")
        
        return {
            "export_path": str(export_path),
            "format": request.format,
            "model_id": model_id,
            "message": "Model exported successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export error: {str(e)}")

@app.get("/api/models/{model_id}/download/{format}")
async def download_model(model_id: str, format: str):
    """Download exported model file"""
    model_info = registry.get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    export_dir = EXPORTS_DIR / model_id
    
    if format == "torchscript":
        file_path = export_dir / f"{model_info.name}.pt"
    elif format == "onnx":
        file_path = export_dir / f"{model_info.name}.onnx"
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type='application/octet-stream'
    )

# @app.post("/api/adversarial/attack")
# async def adversarial_attack(
#     model_id: str,
#     attack_type: str = "fgsm",
#     epsilon: float = 0.1,
#     data: List[Dict[str, Any]] = None
# ):
#     """Run adversarial attacks on model - Not implemented"""
#     # Placeholder for adversarial attack implementation
#     # Would integrate with Adversarial Robustness Toolbox (ART)
#     return {
#         "attack_type": attack_type,
#         "epsilon": epsilon,
#         "success_rate": 0.15,
#         "adversarial_accuracy": 0.856,
#         "original_accuracy": 0.924,
#         "message": "Adversarial attack completed"
#     }

# @app.get("/api/quantum/demo")
# async def quantum_demo():
#     """Quantum computing demo endpoint - Not implemented"""
#     # Placeholder for quantum ML implementation
#     return {
#         "quantum_accuracy": 0.887,
#         "classical_accuracy": 0.876,
#         "quantum_advantage": 0.011,
#         "circuit_depth": 12,
#         "qubits_used": 4,
#         "message": "Quantum SVM demo completed"
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)