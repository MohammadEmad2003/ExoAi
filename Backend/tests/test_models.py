"""
Unit tests for model classes
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.tabkanet import TabKANet, KANLayer


class TestKANLayer:
    """Test KANLayer class"""
    
    def test_kan_layer_creation(self):
        """Test KANLayer creation with valid parameters"""
        layer = KANLayer(n_inputs=10, n_out=5, K_inner=16, d_model=64)
        assert layer.n_inputs == 10
        assert layer.n_out == 5
        assert layer.K == 16
        assert layer.d_model == 64
    
    def test_kan_layer_forward(self):
        """Test KANLayer forward pass"""
        layer = KANLayer(n_inputs=5, n_out=3, K_inner=8, d_model=32)
        x = torch.randn(2, 5)
        output = layer(x)
        
        assert output.shape == (2, 3, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_kan_layer_gradients(self):
        """Test that KANLayer produces gradients"""
        layer = KANLayer(n_inputs=4, n_out=2, K_inner=8, d_model=16)
        x = torch.randn(3, 4, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTabKANet:
    """Test TabKANet class"""
    
    def test_tabkanet_creation(self):
        """Test TabKANet creation with valid parameters"""
        model = TabKANet(
            n_num=10, n_cat=5, cat_card_list=[2, 3, 4, 2, 3],
            d_model=64, K_inner=16, trans_heads=4, trans_depth=2,
            mlp_hidden=128, n_classes=3, dropout=0.1
        )
        
        assert model.n_num == 10
        assert model.n_cat == 5
        assert model.d_model == 64
        assert model.n_classes == 3
    
    def test_tabkanet_forward_numeric_only(self):
        """Test TabKANet forward pass with numeric features only"""
        model = TabKANet(
            n_num=5, n_cat=0, cat_card_list=[],
            d_model=32, K_inner=8, trans_heads=2, trans_depth=1,
            mlp_hidden=64, n_classes=2, dropout=0.0
        )
        
        x_num = torch.randn(3, 5)
        x_cat = torch.zeros(3, 0, dtype=torch.long)
        output = model(x_num, x_cat)
        
        assert output.shape == (3, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_tabkanet_forward_categorical_only(self):
        """Test TabKANet forward pass with categorical features only"""
        model = TabKANet(
            n_num=0, n_cat=3, cat_card_list=[2, 3, 4],
            d_model=32, K_inner=8, trans_heads=2, trans_depth=1,
            mlp_hidden=64, n_classes=2, dropout=0.0
        )
        
        x_num = torch.zeros(3, 0)
        x_cat = torch.randint(0, 2, (3, 3))
        output = model(x_num, x_cat)
        
        assert output.shape == (3, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_tabkanet_forward_mixed(self):
        """Test TabKANet forward pass with both numeric and categorical features"""
        model = TabKANet(
            n_num=5, n_cat=3, cat_card_list=[2, 3, 4],
            d_model=32, K_inner=8, trans_heads=2, trans_depth=1,
            mlp_hidden=64, n_classes=3, dropout=0.0
        )
        
        x_num = torch.randn(4, 5)
        x_cat = torch.randint(0, 2, (4, 3))
        output = model(x_num, x_cat)
        
        assert output.shape == (4, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_tabkanet_gradients(self):
        """Test that TabKANet produces gradients"""
        model = TabKANet(
            n_num=3, n_cat=2, cat_card_list=[2, 3],
            d_model=16, K_inner=4, trans_heads=2, trans_depth=1,
            mlp_hidden=32, n_classes=2, dropout=0.0
        )
        
        x_num = torch.randn(2, 3, requires_grad=True)
        x_cat = torch.randint(0, 2, (2, 2))
        output = model(x_num, x_cat)
        loss = output.sum()
        loss.backward()
        
        assert x_num.grad is not None
        assert not torch.isnan(x_num.grad).any()
    
    def test_tabkanet_parameter_count(self):
        """Test that TabKANet has reasonable number of parameters"""
        model = TabKANet(
            n_num=10, n_cat=5, cat_card_list=[2, 3, 4, 2, 3],
            d_model=64, K_inner=16, trans_heads=4, trans_depth=2,
            mlp_hidden=128, n_classes=3, dropout=0.1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1000  # Should have reasonable number of parameters
        assert total_params < 1000000  # But not too many


class TestModelIntegration:
    """Integration tests for model components"""
    
    def test_model_training_step(self):
        """Test a single training step"""
        model = TabKANet(
            n_num=5, n_cat=2, cat_card_list=[2, 3],
            d_model=32, K_inner=8, trans_heads=2, trans_depth=1,
            mlp_hidden=64, n_classes=3, dropout=0.0
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create dummy data
        x_num = torch.randn(4, 5)
        x_cat = torch.randint(0, 2, (4, 2))
        y = torch.randint(0, 3, (4,))
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x_num, x_cat)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss).any()
        assert loss.item() > 0
    
    def test_model_inference_mode(self):
        """Test model in inference mode"""
        model = TabKANet(
            n_num=3, n_cat=2, cat_card_list=[2, 3],
            d_model=16, K_inner=4, trans_heads=2, trans_depth=1,
            mlp_hidden=32, n_classes=2, dropout=0.0
        )
        
        model.eval()
        
        x_num = torch.randn(2, 3)
        x_cat = torch.randint(0, 2, (2, 2))
        
        with torch.no_grad():
            output = model(x_num, x_cat)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
        
        assert output.shape == (2, 2)
        assert probabilities.shape == (2, 2)
        assert predictions.shape == (2,)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
