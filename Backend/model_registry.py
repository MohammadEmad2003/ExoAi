"""
Simple model registry for storing and managing trained models.
Provides metadata storage, versioning, and model lifecycle management.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

import torch
import pandas as pd
from pydantic import BaseModel

class ModelMetadata(BaseModel):
    """Model metadata schema"""
    id: str
    name: str
    version: str
    description: Optional[str] = ""
    architecture: str
    framework: str = "PyTorch"
    created_at: str
    updated_at: str
    created_by: str = "system"
    
    # Training info
    dataset_name: Optional[str] = None
    dataset_size: Optional[int] = None
    target_column: Optional[str] = None
    feature_count: Optional[int] = None
    
    # Model info
    parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    training_time: Optional[str] = None
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Paths
    model_path: str
    config_path: Optional[str] = None
    
    # Tags and labels
    tags: List[str] = []
    labels: Dict[str, str] = {}
    
    # Status
    status: str = "active"  # active, archived, deprecated
    
    class Config:
        extra = "allow"

class ModelRegistry:
    """Simple file-based model registry"""
    
    def __init__(self, registry_dir: str = "models_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.exports_dir = self.registry_dir / "exports"
        
        for dir_path in [self.models_dir, self.metadata_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load existing registry
        self.registry_file = self.registry_dir / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        name: str,
        architecture: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a new model"""
        
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Copy model file to registry
        source_path = Path(model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Copy model file
        registry_model_path = model_dir / source_path.name
        shutil.copy2(source_path, registry_model_path)
        
        # Get model size
        model_size_mb = registry_model_path.stat().st_size / (1024 * 1024)
        
        # Create metadata
        now = datetime.now().isoformat()
        
        # Try to extract additional info from model checkpoint
        model_info = {}
        if registry_model_path.suffix == '.pt':
            try:
                checkpoint = torch.load(registry_model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    model_info = {
                        'parameters': sum(p.numel() for p in checkpoint.get('model_state_dict', {}).values() if hasattr(p, 'numel')),
                        'dataset_size': checkpoint.get('dataset_size'),
                        'target_column': checkpoint.get('target_column'),
                        'feature_count': checkpoint.get('n_features'),
                        'final_metrics': checkpoint.get('final_metrics', {})
                    }
            except Exception as e:
                print(f"Warning: Could not extract model info: {e}")
        
        # Merge provided metadata
        if metadata:
            model_info.update(metadata)
        
        # Extract metrics
        final_metrics = model_info.get('final_metrics', {})
        
        # Create model metadata
        model_metadata = ModelMetadata(
            id=model_id,
            name=name,
            version="1.0.0",
            architecture=architecture,
            created_at=now,
            updated_at=now,
            model_path=str(registry_model_path),
            model_size_mb=model_size_mb,
            parameters=model_info.get('parameters'),
            dataset_size=model_info.get('dataset_size'),
            target_column=model_info.get('target_column'),
            feature_count=model_info.get('feature_count'),
            accuracy=final_metrics.get('accuracy'),
            precision=final_metrics.get('precision'),
            recall=final_metrics.get('recall'),
            f1_score=final_metrics.get('f1_score'),
            auc_roc=final_metrics.get('auc_roc')
        )
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata.dict(), f, indent=2)
        
        # Update registry
        self.registry[model_id] = model_metadata.dict()
        self._save_registry()
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        if model_id not in self.registry:
            return None
        
        return ModelMetadata(**self.registry[model_id])
    
    def list_models(
        self,
        status: Optional[str] = None,
        architecture: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering"""
        
        models = []
        for model_data in self.registry.values():
            model = ModelMetadata(**model_data)
            
            # Apply filters
            if status and model.status != status:
                continue
            
            if architecture and model.architecture != architecture:
                continue
            
            if tags and not any(tag in model.tags for tag in tags):
                continue
            
            models.append(model)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update model metadata"""
        if model_id not in self.registry:
            return False
        
        # Update registry
        self.registry[model_id].update(updates)
        self.registry[model_id]['updated_at'] = datetime.now().isoformat()
        
        # Update metadata file
        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.registry[model_id], f, indent=2)
        
        self._save_registry()
        return True
    
    def delete_model(self, model_id: str, remove_files: bool = True) -> bool:
        """Delete model from registry"""
        if model_id not in self.registry:
            return False
        
        if remove_files:
            # Remove model directory
            model_dir = self.models_dir / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove metadata file
            metadata_path = self.metadata_dir / f"{model_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
        
        # Remove from registry
        del self.registry[model_id]
        self._save_registry()
        
        return True
    
    def archive_model(self, model_id: str) -> bool:
        """Archive a model (mark as archived)"""
        return self.update_model(model_id, {'status': 'archived'})
    
    def add_tags(self, model_id: str, tags: List[str]) -> bool:
        """Add tags to a model"""
        if model_id not in self.registry:
            return False
        
        current_tags = set(self.registry[model_id].get('tags', []))
        current_tags.update(tags)
        
        return self.update_model(model_id, {'tags': list(current_tags)})
    
    def remove_tags(self, model_id: str, tags: List[str]) -> bool:
        """Remove tags from a model"""
        if model_id not in self.registry:
            return False
        
        current_tags = set(self.registry[model_id].get('tags', []))
        current_tags.difference_update(tags)
        
        return self.update_model(model_id, {'tags': list(current_tags)})
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """Search models by name, description, or tags"""
        query = query.lower()
        results = []
        
        for model_data in self.registry.values():
            model = ModelMetadata(**model_data)
            
            # Search in name, description, and tags
            searchable_text = f"{model.name} {model.description} {' '.join(model.tags)}".lower()
            
            if query in searchable_text:
                results.append(model)
        
        return results
    
    def get_model_versions(self, name: str) -> List[ModelMetadata]:
        """Get all versions of a model by name"""
        versions = []
        
        for model_data in self.registry.values():
            model = ModelMetadata(**model_data)
            if model.name == name:
                versions.append(model)
        
        # Sort by version
        versions.sort(key=lambda x: x.version, reverse=True)
        return versions
    
    def create_model_version(
        self,
        base_model_id: str,
        new_model_path: str,
        version: str = None
    ) -> str:
        """Create a new version of an existing model"""
        
        base_model = self.get_model(base_model_id)
        if not base_model:
            raise ValueError(f"Base model {base_model_id} not found")
        
        # Auto-increment version if not provided
        if not version:
            versions = self.get_model_versions(base_model.name)
            if versions:
                # Simple version increment (assumes semantic versioning)
                latest_version = versions[0].version
                major, minor, patch = map(int, latest_version.split('.'))
                version = f"{major}.{minor}.{patch + 1}"
            else:
                version = "1.0.1"
        
        # Register new version
        return self.register_model(
            model_path=new_model_path,
            name=base_model.name,
            architecture=base_model.architecture,
            metadata={
                'version': version,
                'description': f"Version {version} of {base_model.name}",
                'tags': base_model.tags,
                'labels': base_model.labels
            }
        )
    
    def export_model(self, model_id: str, export_format: str = "torchscript") -> str:
        """Export model to specified format"""
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Create export directory
        export_dir = self.exports_dir / model_id
        export_dir.mkdir(exist_ok=True)
        
        if export_format == "torchscript":
            # Load and convert to TorchScript
            checkpoint = torch.load(model.model_path, map_location='cpu')
            
            # This would need to be implemented based on the specific model architecture
            # For now, just copy the original file
            export_path = export_dir / f"{model.name}_v{model.version}.pt"
            shutil.copy2(model.model_path, export_path)
            
        elif export_format == "onnx":
            # ONNX export would be implemented here
            export_path = export_dir / f"{model.name}_v{model.version}.onnx"
            # Placeholder - actual ONNX conversion needed
            
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return str(export_path)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        models = list(self.registry.values())
        
        stats = {
            'total_models': len(models),
            'active_models': len([m for m in models if m.get('status') == 'active']),
            'archived_models': len([m for m in models if m.get('status') == 'archived']),
            'architectures': {},
            'total_size_mb': 0,
            'avg_accuracy': 0
        }
        
        # Calculate architecture distribution
        for model in models:
            arch = model.get('architecture', 'unknown')
            stats['architectures'][arch] = stats['architectures'].get(arch, 0) + 1
        
        # Calculate total size and average accuracy
        accuracies = []
        for model in models:
            if model.get('model_size_mb'):
                stats['total_size_mb'] += model['model_size_mb']
            
            if model.get('accuracy'):
                accuracies.append(model['accuracy'])
        
        if accuracies:
            stats['avg_accuracy'] = sum(accuracies) / len(accuracies)
        
        return stats

# Global registry instance
model_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    return model_registry
