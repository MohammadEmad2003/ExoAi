"""
Model loading factory for dynamic model instantiation
"""

import importlib
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_model_from_config(config_path: str, checkpoint_path: Optional[str] = None, 
                          device: str = "auto") -> torch.nn.Module:
    """
    Load a model from configuration file and optional checkpoint
    
    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to load model on ("auto", "cpu", "cuda")
    
    Returns:
        Loaded PyTorch model
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get model class and constructor args
    model_class_path = config['model']['class']
    model_args = config['model']['constructor']
    
    # Import model class dynamically
    module_path, class_name = model_class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Create model instance
    model = model_class(**model_args)
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    return model


def load_model_from_checkpoint(checkpoint_path: str, model_class_path: str, 
                              model_args: Dict[str, Any], device: str = "auto") -> torch.nn.Module:
    """
    Load a model from checkpoint with explicit class specification
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_class_path: Python path to model class (e.g., "models.tabkanet.TabKANet")
        model_args: Constructor arguments for model
        device: Device to load model on
    
    Returns:
        Loaded PyTorch model
    """
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Import model class dynamically
    module_path, class_name = model_class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Create model instance
    model = model_class(**model_args)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    return model


def get_model_info(config_path: str) -> Dict[str, Any]:
    """
    Get model information from configuration file
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary with model information
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return {
        'class': config['model']['class'],
        'constructor': config['model']['constructor'],
        'input_shapes': {
            'numeric': config['input']['numeric_shape'],
            'categorical': config['input']['categorical_shape']
        },
        'dtype': config['input']['dtype'],
        'device': config['input']['device']
    }
