#!/usr/bin/env python3
"""
Model Export CLI Script
Converts PyTorch models to production-ready inference formats
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from load_model import load_model_from_config, get_model_info


def export_torchscript(model: torch.nn.Module, example_inputs: Dict[str, torch.Tensor], 
                      output_path: str, optimize: bool = True) -> str:
    """Export model to TorchScript format"""
    model.eval()
    
    try:
        # Try tracing first (preferred for most models)
        traced_model = torch.jit.trace(model, (example_inputs['x_num'], example_inputs['x_cat']))
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save traced model
        traced_path = output_path.replace('.pt', '_traced.pt')
        traced_model.save(traced_path)
        print(f"✓ TorchScript (traced) saved to: {traced_path}")
        
        # Also try scripting for better optimization
        try:
            scripted_model = torch.jit.script(model)
            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            scripted_path = output_path.replace('.pt', '_scripted.pt')
            scripted_model.save(scripted_path)
            print(f"✓ TorchScript (scripted) saved to: {scripted_path}")
            return scripted_path
        except Exception as e:
            print(f"⚠ Scripting failed, using traced model: {e}")
            return traced_path
            
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        raise


def export_onnx(model: torch.nn.Module, example_inputs: Dict[str, torch.Tensor], 
               output_path: str, opset: int = 13, dynamic_axes: Optional[Dict] = None) -> str:
    """Export model to ONNX format"""
    model.eval()
    
    # Prepare input names and shapes
    input_names = ['x_num', 'x_cat']
    output_names = ['logits']
    
    # Create example inputs tuple
    example_inputs_tuple = (example_inputs['x_num'], example_inputs['x_cat'])
    
    try:
        torch.onnx.export(
            model,
            example_inputs_tuple,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"✓ ONNX model saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        raise


def export_tensorrt(model: torch.nn.Module, example_inputs: Dict[str, torch.Tensor], 
                   output_path: str, max_batch_size: int = 32) -> str:
    """Export model to TensorRT format (requires TensorRT)"""
    try:
        import tensorrt as trt
        import torch2trt
    except ImportError:
        raise ImportError("TensorRT and torch2trt are required for TensorRT export")
    
    model.eval()
    
    try:
        # Convert to TensorRT
        example_inputs_tuple = (example_inputs['x_num'], example_inputs['x_cat'])
        trt_model = torch2trt.torch2trt(
            model, 
            example_inputs_tuple,
            max_batch_size=max_batch_size,
            fp16_mode=True,  # Use FP16 for better performance
            strict_type_constraints=True
        )
        
        # Save TensorRT engine
        torch.save(trt_model.state_dict(), output_path)
        print(f"✓ TensorRT model saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ TensorRT export failed: {e}")
        raise


def validate_exported_model(original_model: torch.nn.Module, exported_path: str, 
                          example_inputs: Dict[str, torch.Tensor], format_type: str,
                          tolerance: float = 1e-5) -> bool:
    """Validate exported model against original"""
    print(f"Validating {format_type} model...")
    
    # Get original model predictions
    original_model.eval()
    with torch.no_grad():
        original_output = original_model(example_inputs['x_num'], example_inputs['x_cat'])
    
    try:
        if format_type == "torchscript":
            # Load and test TorchScript model
            exported_model = torch.jit.load(exported_path)
            exported_model.eval()
            with torch.no_grad():
                exported_output = exported_model(example_inputs['x_num'], example_inputs['x_cat'])
                
        elif format_type == "onnx":
            # Test ONNX model with ONNX Runtime
            try:
                import onnxruntime as ort
            except ImportError:
                print("⚠ ONNX Runtime not available, skipping validation")
                return True
            
            # Prepare inputs for ONNX Runtime
            ort_inputs = {
                'x_num': example_inputs['x_num'].cpu().numpy(),
                'x_cat': example_inputs['x_cat'].cpu().numpy()
            }
            
            # Run inference
            ort_session = ort.InferenceSession(exported_path)
            exported_output = ort_session.run(None, ort_inputs)[0]
            exported_output = torch.from_numpy(exported_output)
            
        else:
            print(f"⚠ Validation not implemented for {format_type}")
            return True
        
        # Compare outputs
        if isinstance(exported_output, torch.Tensor):
            max_diff = torch.max(torch.abs(original_output - exported_output)).item()
            mean_diff = torch.mean(torch.abs(original_output - exported_output)).item()
        else:
            max_diff = np.max(np.abs(original_output.cpu().numpy() - exported_output))
            mean_diff = np.mean(np.abs(original_output.cpu().numpy() - exported_output))
        
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        if max_diff < tolerance:
            print(f"✓ {format_type} model validation passed")
            return True
        else:
            print(f"✗ {format_type} model validation failed (tolerance: {tolerance})")
            return False
            
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def create_example_inputs(config: Dict[str, Any], batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """Create example inputs for model export"""
    numeric_shape = [batch_size] + config['input']['numeric_shape']
    categorical_shape = [batch_size] + config['input']['categorical_shape']
    
    # Create random inputs with appropriate shapes
    x_num = torch.randn(numeric_shape, dtype=torch.float32)
    x_cat = torch.randint(0, 2, categorical_shape, dtype=torch.long)  # Binary categorical for simplicity
    
    return {'x_num': x_num, 'x_cat': x_cat}


def export_model_cli(config_path: str, checkpoint_path: Optional[str] = None,
                     output_dir: str = './exports', formats: Optional[List[str]] = None,
                     opset: int = 13, batch_size: int = 1, tolerance: float = 1e-5,
                     device: str = 'auto', validate: bool = False) -> Dict[str, Any]:
    """Programmatic export entry point (used by API).

    Returns a dictionary with export results.
    """
    if formats is None:
        formats = ['torchscript', 'onnx']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model = load_model_from_config(config_path, checkpoint_path, device)
    model_device = next(model.parameters()).device

    # Create example inputs and move to device
    example_inputs = create_example_inputs(config, batch_size)
    example_inputs = {k: v.to(model_device) for k, v in example_inputs.items()}

    results: Dict[str, Any] = {}

    for format_type in formats:
        try:
            if format_type == 'torchscript':
                out_path = output_dir / 'model.pt'
                exported_path = export_torchscript(model, example_inputs, str(out_path))
            elif format_type == 'onnx':
                out_path = output_dir / 'model.onnx'
                dynamic_axes = config.get('export', {}).get('dynamic_axes', None)
                exported_path = export_onnx(model, example_inputs, str(out_path), opset, dynamic_axes)
            elif format_type == 'tensorrt':
                out_path = output_dir / 'model_trt.pt'
                exported_path = export_tensorrt(model, example_inputs, str(out_path))
            else:
                results[format_type] = {'error': f'unsupported format: {format_type}'}
                continue

            # Validate if requested
            validation_passed = None
            if validate:
                validation_passed = validate_exported_model(model, exported_path, example_inputs, format_type, tolerance)

            results[format_type] = {
                'path': str(exported_path),
                'validation_passed': validation_passed
            }

        except Exception as e:
            results[format_type] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to production formats')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./exports', help='Output directory for exported models')
    parser.add_argument('--formats', nargs='+', choices=['torchscript', 'onnx', 'tensorrt'], 
                       default=['torchscript', 'onnx'], help='Export formats')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--batch-size', type=int, default=1, help='Example batch size for export')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Validation tolerance')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use for export')
    parser.add_argument('--validate', action='store_true', help='Validate exported models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading model from config: {args.config}")
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load model
    model = load_model_from_config(args.config, args.checkpoint, args.device)
    print(f"Model loaded on device: {next(model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create example inputs
    example_inputs = create_example_inputs(config, args.batch_size)
    example_inputs = {k: v.to(next(model.parameters()).device) for k, v in example_inputs.items()}
    
    print(f"Example input shapes: {[(k, v.shape) for k, v in example_inputs.items()]}")
    
    # Export to requested formats
    results = {}
    
    for format_type in args.formats:
        print(f"\n{'='*50}")
        print(f"Exporting to {format_type.upper()}")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            if format_type == "torchscript":
                output_path = output_dir / "model.pt"
                exported_path = export_torchscript(model, example_inputs, str(output_path))
                
            elif format_type == "onnx":
                output_path = output_dir / "model.onnx"
                dynamic_axes = config.get('export', {}).get('dynamic_axes', None)
                exported_path = export_onnx(model, example_inputs, str(output_path), 
                                         args.opset, dynamic_axes)
                
            elif format_type == "tensorrt":
                output_path = output_dir / "model_trt.pt"
                exported_path = export_tensorrt(model, example_inputs, str(output_path))
            
            export_time = time.time() - start_time
            print(f"Export completed in {export_time:.2f}s")
            
            # Validate if requested
            if args.validate:
                validation_passed = validate_exported_model(
                    model, exported_path, example_inputs, format_type, args.tolerance
                )
                results[format_type] = {
                    'path': exported_path,
                    'export_time': export_time,
                    'validation_passed': validation_passed
                }
            else:
                results[format_type] = {
                    'path': exported_path,
                    'export_time': export_time,
                    'validation_passed': None
                }
                
        except Exception as e:
            print(f"✗ Export to {format_type} failed: {e}")
            results[format_type] = {'error': str(e)}
    
    # Save export results
    results_path = output_dir / "export_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print("EXPORT SUMMARY")
    print(f"{'='*50}")
    
    for format_type, result in results.items():
        if 'error' in result:
            print(f"❌ {format_type.upper()}: FAILED - {result['error']}")
        else:
            status = "✅ PASSED" if result.get('validation_passed', True) else "⚠️  FAILED VALIDATION"
            print(f"✅ {format_type.upper()}: {result['path']} ({result['export_time']:.2f}s) {status}")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Export directory: {output_dir}")


if __name__ == "__main__":
    main()
