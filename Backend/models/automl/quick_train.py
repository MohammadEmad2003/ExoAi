"""
Quick training module using LightGBM for baseline models.
Provides automated model training with minimal configuration for small datasets.
"""

import os
import json
import logging
import pickle
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import time
import uuid

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import lightgbm as lgb

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for quick training."""
    task_type: str  # classification, regression
    target_column: str
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    max_training_time: int = 300  # 5 minutes max
    early_stopping_rounds: int = 50

@dataclass
class ModelMetrics:
    """Training and validation metrics."""
    task_type: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    feature_importance: Dict[str, float]

@dataclass
class TrainingResult:
    """Complete training result."""
    model_id: str
    dataset_id: str
    config: TrainingConfig
    metrics: ModelMetrics
    model_path: str
    preprocessing_path: str
    training_time: float
    status: str
    created_at: float

class LightGBMQuickTrainer:
    """Quick trainer using LightGBM with automated preprocessing."""
    
    def __init__(self, 
                 models_dir: str = "models/trained",
                 artifacts_dir: str = "exports"):
        """
        Initialize quick trainer.
        
        Args:
            models_dir: Directory to save trained models
            artifacts_dir: Directory to save model artifacts
        """
        self.models_dir = models_dir
        self.artifacts_dir = artifacts_dir
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        
        logger.info(f"Initialized LightGBMQuickTrainer (models_dir={models_dir})")
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect task type based on target variable."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        
        # For numeric targets, check if it looks like classification
        unique_values = y.nunique()
        unique_ratio = unique_values / len(y)
        
        if unique_values <= 20 and unique_ratio < 0.05:
            return 'classification'
        
        return 'regression'
    
    def _create_preprocessor(self, X: pd.DataFrame, y: pd.Series) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove high-cardinality categorical features
        filtered_categorical = []
        for col in categorical_features:
            cardinality = X[col].nunique()
            if cardinality <= 50:  # Reasonable cardinality threshold
                filtered_categorical.append(col)
            else:
                logger.warning(f"Dropping high-cardinality feature: {col} (cardinality: {cardinality})")
        
        categorical_features = filtered_categorical
        
        # Create transformers
        transformers = []
        
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        
        if categorical_features:
            # Use OneHotEncoder with handle_unknown='ignore' for robustness
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
        
        if not transformers:
            raise ValueError("No valid features found for training")
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any remaining columns
        )
        
        return preprocessor
    
    def _get_lgb_params(self, task_type: str, n_samples: int) -> Dict[str, Any]:
        """Get LightGBM parameters based on task and data size."""
        
        base_params = {
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
            'force_col_wise': True,  # Better for small datasets
        }
        
        if task_type == 'classification':
            base_params.update({
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_class': None,  # Will be set during training
            })
        else:  # regression
            base_params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
            })
        
        # Adjust parameters based on dataset size
        if n_samples < 1000:
            base_params.update({
                'num_leaves': 15,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 10,
            })
        elif n_samples < 10000:
            base_params.update({
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
            })
        else:
            base_params.update({
                'num_leaves': 63,
                'learning_rate': 0.03,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 50,
            })
        
        return base_params
    
    def _calculate_metrics(self, 
                          task_type: str, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate metrics based on task type."""
        
        metrics = {}
        
        if task_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2 and y_prob is not None:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
                except:
                    pass
        
        else:  # regression
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    async def train_model(self,
                         df: pd.DataFrame,
                         config: TrainingConfig,
                         dataset_id: str) -> TrainingResult:
        """
        Train a quick baseline model.
        
        Args:
            df: Training dataset
            config: Training configuration
            dataset_id: Dataset identifier
            
        Returns:
            Training result with metrics and model paths
        """
        start_time = time.time()
        model_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting quick training for dataset {dataset_id}")
            
            # Prepare data
            if config.target_column not in df.columns:
                raise ValueError(f"Target column '{config.target_column}' not found in dataset")
            
            X = df.drop(columns=[config.target_column])
            y = df[config.target_column]
            
            # Remove rows with missing targets
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                raise ValueError("No valid samples after removing missing targets")
            
            # Detect task type if not specified
            task_type = config.task_type
            if task_type == 'auto':
                task_type = self._detect_task_type(y)
                config.task_type = task_type
            
            logger.info(f"Detected task type: {task_type}")
            
            # Encode target for classification
            target_encoder = None
            if task_type == 'classification' and y.dtype == 'object':
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y)
                y = pd.Series(y_encoded, index=y.index)
            
            # Create preprocessor
            preprocessor = self._create_preprocessor(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.test_size, 
                random_state=config.random_state,
                stratify=y if task_type == 'classification' else None
            )
            
            # Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Get LightGBM parameters
            lgb_params = self._get_lgb_params(task_type, len(X_train))
            
            # Set num_class for multiclass classification
            if task_type == 'classification':
                lgb_params['num_class'] = len(np.unique(y))
                if lgb_params['num_class'] == 2:
                    lgb_params['objective'] = 'binary'
                    lgb_params['metric'] = 'binary_logloss'
                    del lgb_params['num_class']
            
            # Create datasets
            train_data = lgb.Dataset(X_train_processed, label=y_train)
            valid_data = lgb.Dataset(X_test_processed, label=y_test, reference=train_data)
            
            # Train model
            logger.info("Training LightGBM model...")
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(config.early_stopping_rounds),
                    lgb.log_evaluation(period=0)  # Suppress verbose output
                ]
            )
            
            # Make predictions
            y_train_pred = model.predict(X_train_processed)
            y_test_pred = model.predict(X_test_processed)
            
            # Handle prediction format for classification
            if task_type == 'classification':
                if len(np.unique(y)) == 2:
                    # Binary classification
                    y_train_prob = y_train_pred.reshape(-1, 1)
                    y_test_prob = y_test_pred.reshape(-1, 1)
                    y_train_pred = (y_train_pred > 0.5).astype(int)
                    y_test_pred = (y_test_pred > 0.5).astype(int)
                    # Create probability matrix for metrics
                    y_train_prob = np.column_stack([1 - y_train_prob, y_train_prob])
                    y_test_prob = np.column_stack([1 - y_test_prob, y_test_prob])
                else:
                    # Multiclass classification
                    y_train_prob = y_train_pred
                    y_test_prob = y_test_pred
                    y_train_pred = np.argmax(y_train_pred, axis=1)
                    y_test_pred = np.argmax(y_test_pred, axis=1)
            else:
                y_train_prob = None
                y_test_prob = None
            
            # Calculate metrics
            training_metrics = self._calculate_metrics(task_type, y_train, y_train_pred, y_train_prob)
            validation_metrics = self._calculate_metrics(task_type, y_test, y_test_pred, y_test_prob)
            
            # Cross-validation
            cv_scores = {}
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
                scoring = 'neg_mean_squared_error'
            
            try:
                # Create a pipeline for CV
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', lgb.LGBMRegressor(**lgb_params) if task_type == 'regression' 
                     else lgb.LGBMClassifier(**lgb_params))
                ])
                
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
                cv_scores[scoring] = scores.tolist()
                
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                cv_scores = {}
            
            # Feature importance
            feature_names = []
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out().tolist()
            else:
                # Fallback for older sklearn versions
                numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                feature_names = numeric_features + categorical_features
            
            importance_scores = model.feature_importance(importance_type='gain')
            feature_importance = {}
            
            for i, score in enumerate(importance_scores):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(score)
            
            # Save model and preprocessor
            model_filename = f"{model_id}_model.pkl"
            preprocessor_filename = f"{model_id}_preprocessor.pkl"
            
            model_path = os.path.join(self.models_dir, model_filename)
            preprocessor_path = os.path.join(self.models_dir, preprocessor_filename)
            
            # Save artifacts
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(preprocessor_path, 'wb') as f:
                pickle.dump({
                    'preprocessor': preprocessor,
                    'target_encoder': target_encoder,
                    'feature_names': feature_names
                }, f)
            
            # Create result
            metrics = ModelMetrics(
                task_type=task_type,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                cv_scores=cv_scores,
                feature_importance=feature_importance
            )
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                model_id=model_id,
                dataset_id=dataset_id,
                config=config,
                metrics=metrics,
                model_path=model_path,
                preprocessing_path=preprocessor_path,
                training_time=training_time,
                status="completed",
                created_at=time.time()
            )
            
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Validation metrics: {validation_metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Return failed result
            return TrainingResult(
                model_id=model_id,
                dataset_id=dataset_id,
                config=config,
                metrics=ModelMetrics(
                    task_type=config.task_type,
                    training_metrics={},
                    validation_metrics={},
                    cv_scores={},
                    feature_importance={}
                ),
                model_path="",
                preprocessing_path="",
                training_time=time.time() - start_time,
                status=f"failed: {str(e)}",
                created_at=time.time()
            )
    
    def load_model(self, model_path: str, preprocessor_path: str) -> Tuple[Any, Any, Optional[Any]]:
        """Load trained model and preprocessor."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(preprocessor_path, 'rb') as f:
                preprocessing_data = pickle.load(f)
                preprocessor = preprocessing_data['preprocessor']
                target_encoder = preprocessing_data.get('target_encoder')
            
            return model, preprocessor, target_encoder
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, 
               model_path: str,
               preprocessor_path: str,
               X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using trained model."""
        try:
            model, preprocessor, target_encoder = self.load_model(model_path, preprocessor_path)
            
            # Preprocess input
            X_processed = preprocessor.transform(X)
            
            # Make predictions
            predictions = model.predict(X_processed)
            
            # Handle classification predictions
            if target_encoder is not None:
                predictions = target_encoder.inverse_transform(predictions.astype(int))
            
            return {
                "predictions": predictions.tolist(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "predictions": [],
                "status": f"failed: {str(e)}"
            }

# Factory function
def create_quick_trainer(**kwargs) -> LightGBMQuickTrainer:
    """Create quick trainer with custom parameters."""
    return LightGBMQuickTrainer(**kwargs)
