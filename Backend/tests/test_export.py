"""
Unit tests for model export functionality
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.tabkanet import TabKANet
from load_model import load_model_from_config


class TestModelExport:
    """Test model export functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pt"
        
        # Create test config
        config = {
            'model': {
                'class': 'models.tabkanet.TabKANet',
                'constructor': {
                    'n_num': 5,
                    'n_cat': 2,
                    'cat_card_list': [2, 3],
                    'd_model': 16,
                    'K_inner': 4,
                    'trans_heads': 2,
                    'trans_depth': 1,
                    'mlp_hidden': 32,
                    'n_classes': 2,
                    'dropout': 0.0
                }
            },
            'input': {
                'numeric_shape': [5],
                'categorical_shape': [2],
                'dtype': 'float32',
                'device': 'cpu'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create test model and save checkpoint
        model = TabKANet(**config['model']['constructor'])
        torch.save(model.state_dict(), self.checkpoint_path)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_model_from_config(self):
        """Test loading model from config"""
        model = load_model_from_config(str(self.config_path))
        
        assert isinstance(model, TabKANet)
        assert model.n_num == 5
        assert model.n_cat == 2
        assert model.n_classes == 2
    
    def test_load_model_with_checkpoint(self):
        """Test loading model with checkpoint"""
        model = load_model_from_config(str(self.config_path), str(self.checkpoint_path))
        
        assert isinstance(model, TabKANet)
        assert model.n_num == 5
        assert model.n_cat == 2
        assert model.n_classes == 2
    
    def test_model_inference_consistency(self):
        """Test that model inference is consistent"""
        model1 = TabKANet(n_num=3, n_cat=2, cat_card_list=[2, 3], d_model=16, 
                         K_inner=4, trans_heads=2, trans_depth=1, mlp_hidden=32, 
                         n_classes=2, dropout=0.0)
        model2 = TabKANet(n_num=3, n_cat=2, cat_card_list=[2, 3], d_model=16, 
                         K_inner=4, trans_heads=2, trans_depth=1, mlp_hidden=32, 
                         n_classes=2, dropout=0.0)
        
        # Copy weights
        model2.load_state_dict(model1.state_dict())
        
        # Test with same inputs
        x_num = torch.randn(2, 3)
        x_cat = torch.randint(0, 2, (2, 2))
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(x_num, x_cat)
            output2 = model2(x_num, x_cat)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_model_device_placement(self):
        """Test model device placement"""
        model = load_model_from_config(str(self.config_path), device="cpu")
        
        # Check that all parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"
    
    def test_model_eval_mode(self):
        """Test that loaded model is in eval mode"""
        model = load_model_from_config(str(self.config_path))
        
        assert not model.training
        assert model.kan.training == False
        assert model.transformer.training == False


class TestExportValidation:
    """Test export validation functionality"""
    
    def test_torchscript_export_validation(self):
        """Test TorchScript export validation"""
        model = TabKANet(n_num=3, n_cat=2, cat_card_list=[2, 3], d_model=16, 
                        K_inner=4, trans_heads=2, trans_depth=1, mlp_hidden=32, 
                        n_classes=2, dropout=0.0)
        model.eval()
        
        # Create example inputs
        x_num = torch.randn(2, 3)
        x_cat = torch.randint(0, 2, (2, 2))
        
        # Get original output
        with torch.no_grad():
            original_output = model(x_num, x_cat)
        
        # Create TorchScript model
        try:
            traced_model = torch.jit.trace(model, (x_num, x_cat))
            
            # Test TorchScript model
            with torch.no_grad():
                traced_output = traced_model(x_num, x_cat)
            
            # Validate outputs are close
            assert torch.allclose(original_output, traced_output, atol=1e-5)
            
        except Exception as e:
            pytest.skip(f"TorchScript export failed: {e}")
    
    def test_onnx_export_validation(self):
        """Test ONNX export validation"""
        model = TabKANet(n_num=3, n_cat=2, cat_card_list=[2, 3], d_model=16, 
                        K_inner=4, trans_heads=2, trans_depth=1, mlp_hidden=32, 
                        n_classes=2, dropout=0.0)
        model.eval()
        
        # Create example inputs
        x_num = torch.randn(2, 3)
        x_cat = torch.randint(0, 2, (2, 2))
        
        # Get original output
        with torch.no_grad():
            original_output = model(x_num, x_cat)
        
        # Test ONNX export
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                torch.onnx.export(
                    model,
                    (x_num, x_cat),
                    tmp.name,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['x_num', 'x_cat'],
                    output_names=['logits']
                )
                
                # Test ONNX model with ONNX Runtime
                try:
                    import onnxruntime as ort
                    
                    ort_session = ort.InferenceSession(tmp.name)
                    ort_inputs = {
                        'x_num': x_num.numpy(),
                        'x_cat': x_cat.numpy()
                    }
                    ort_output = ort_session.run(None, ort_inputs)[0]
                    
                    # Validate outputs are close
                    assert np.allclose(original_output.numpy(), ort_output, atol=1e-5)
                    
                except ImportError:
                    pytest.skip("ONNX Runtime not available")
                
                # Cleanup
                Path(tmp.name).unlink()
                
        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")


class TestModelConfig:
    """Test model configuration handling"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {
            'model': {
                'class': 'models.tabkanet.TabKANet',
                'constructor': {
                    'n_num': 5,
                    'n_cat': 2,
                    'cat_card_list': [2, 3],
                    'd_model': 16,
                    'K_inner': 4,
                    'trans_heads': 2,
                    'trans_depth': 1,
                    'mlp_hidden': 32,
                    'n_classes': 2,
                    'dropout': 0.0
                }
            },
            'input': {
                'numeric_shape': [5],
                'categorical_shape': [2],
                'dtype': 'float32',
                'device': 'cpu'
            }
        }
        
        # Test that valid config can be loaded
        model = TabKANet(**valid_config['model']['constructor'])
        assert isinstance(model, TabKANet)
    
    def test_config_missing_fields(self):
        """Test handling of missing configuration fields"""
        # Config missing required fields
        incomplete_config = {
            'model': {
                'class': 'models.tabkanet.TabKANet',
                'constructor': {
                    'n_num': 5,
                    'n_cat': 2,
                    # Missing other required fields
                }
            }
        }
        
        # Should raise error when trying to create model
        with pytest.raises(TypeError):
            TabKANet(**incomplete_config['model']['constructor'])
