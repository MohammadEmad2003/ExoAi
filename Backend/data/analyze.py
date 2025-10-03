"""
Dataset analyzer that produces DataProfile JSON and ActionPlan.
Handles profiling, schema detection, PII detection, and task recommendation.
"""

import os
import json
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# Optional: ydata-profiling for detailed reports
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    try:
        import pandas_profiling as pp
        ProfileReport = pp.ProfileReport
        PROFILING_AVAILABLE = True
    except ImportError:
        PROFILING_AVAILABLE = False
        warnings.warn("ydata-profiling not available. Install with: pip install ydata-profiling")

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"

class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    cardinality_ratio: float
    suggested_role: str  # target, feature, identifier, timestamp, ignore
    potential_pii: bool
    statistics: Dict[str, Any]
    sample_values: List[Any]

@dataclass  
class DataProfile:
    """Complete dataset profile."""
    dataset_id: str
    filename: str
    shape: Tuple[int, int]
    size_mb: float
    columns: List[ColumnProfile]
    suggested_task_type: TaskType
    candidate_targets: List[str]
    data_quality: DataQuality
    issues: List[str]
    metadata: Dict[str, Any]
    created_at: float

@dataclass
class ActionPlan:
    """Recommended next steps for the dataset."""
    quick_train_recommended: bool
    chunk_embed_recommended: bool
    preprocessing_steps: List[str]
    estimated_training_time: str
    memory_requirements: str
    suggested_models: List[str]
    priority_actions: List[Dict[str, str]]

class DatasetAnalyzer:
    """Analyzes datasets and generates profiles and action plans."""
    
    def __init__(self,
                 small_dataset_threshold_mb: float = 50.0,
                 small_dataset_threshold_rows: int = 100000,
                 pii_detection: bool = True):
        """
        Initialize dataset analyzer.
        
        Args:
            small_dataset_threshold_mb: Size threshold for small datasets
            small_dataset_threshold_rows: Row threshold for small datasets  
            pii_detection: Whether to detect potential PII
        """
        self.small_threshold_mb = small_dataset_threshold_mb
        self.small_threshold_rows = small_dataset_threshold_rows
        self.pii_detection = pii_detection
        
        # PII detection patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[- ]?)?\d{10}|\(\d{3}\)\s*\d{3}[- ]?\d{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        }
        
        logger.info("Initialized DatasetAnalyzer")
    
    def _detect_pii(self, series: pd.Series, column_name: str) -> bool:
        """Detect potential PII in a column."""
        if not self.pii_detection:
            return False
        
        # Check column name patterns
        pii_column_names = ['email', 'phone', 'ssn', 'social', 'credit', 'card', 'id', 'passport']
        if any(name in column_name.lower() for name in pii_column_names):
            return True
        
        # Check data patterns for string columns
        if series.dtype == 'object':
            sample_text = ' '.join(str(series.dropna().head(100).tolist()))
            
            for pii_type, pattern in self.pii_patterns.items():
                if re.search(pattern, sample_text):
                    logger.warning(f"Potential {pii_type} detected in column: {column_name}")
                    return True
        
        return False
    
    def _suggest_column_role(self, 
                           column: str, 
                           series: pd.Series, 
                           df_shape: Tuple[int, int]) -> str:
        """Suggest role for a column (target, feature, identifier, etc.)."""
        
        # Check for common identifier patterns
        if any(term in column.lower() for term in ['id', 'index', 'key', 'uuid']):
            return 'identifier'
        
        # Check for timestamp patterns
        if any(term in column.lower() for term in ['time', 'date', 'timestamp', 'created', 'updated']):
            if pd.api.types.is_datetime64_any_dtype(series) or series.dtype == 'object':
                return 'timestamp'
        
        # Check cardinality for potential targets
        unique_ratio = series.nunique() / len(series)
        
        # Low cardinality numeric or categorical - potential target
        if unique_ratio < 0.1 and series.nunique() < 50:
            # Check if it looks like a classification target
            if series.dtype in ['object', 'category'] or (series.dtype in ['int64', 'float64'] and series.nunique() <= 10):
                return 'target_candidate'
        
        # High cardinality - likely identifier
        if unique_ratio > 0.9:
            return 'identifier'
        
        # Numeric features
        if pd.api.types.is_numeric_dtype(series):
            return 'feature'
        
        # Categorical features
        if series.dtype in ['object', 'category']:
            return 'feature'
        
        return 'feature'
    
    def _get_column_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Generate statistics for a column."""
        stats = {
            'count': len(series),
            'non_null_count': series.count()
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': float(series.mean()) if not series.isna().all() else None,
                'std': float(series.std()) if not series.isna().all() else None,
                'min': float(series.min()) if not series.isna().all() else None,
                'max': float(series.max()) if not series.isna().all() else None,
                'median': float(series.median()) if not series.isna().all() else None,
                'q25': float(series.quantile(0.25)) if not series.isna().all() else None,
                'q75': float(series.quantile(0.75)) if not series.isna().all() else None
            })
        
        if series.dtype == 'object' or series.dtype.name == 'category':
            value_counts = series.value_counts()
            stats.update({
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'unique_values': int(series.nunique())
            })
        
        return stats
    
    def _detect_task_type(self, df: pd.DataFrame, target_candidates: List[str]) -> TaskType:
        """Detect the most likely ML task type."""
        
        # Check for time series patterns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return TaskType.TIME_SERIES
        
        # Check target candidates
        if target_candidates:
            target_col = target_candidates[0]
            target_series = df[target_col]
            
            # Classification indicators
            if target_series.dtype in ['object', 'category']:
                return TaskType.CLASSIFICATION
            
            # Regression indicators for numeric targets
            if pd.api.types.is_numeric_dtype(target_series):
                unique_ratio = target_series.nunique() / len(target_series)
                
                # Low unique ratio suggests classification
                if unique_ratio < 0.05 or target_series.nunique() <= 20:
                    return TaskType.CLASSIFICATION
                else:
                    return TaskType.REGRESSION
        
        # No clear target - might be clustering or anomaly detection
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        
        if numeric_ratio > 0.7:
            return TaskType.CLUSTERING
        
        return TaskType.UNKNOWN
    
    def _assess_data_quality(self, df: pd.DataFrame, column_profiles: List[ColumnProfile]) -> Tuple[DataQuality, List[str]]:
        """Assess overall data quality and identify issues."""
        issues = []
        quality_score = 100
        
        # Check missing data
        total_missing = sum(col.null_percentage for col in column_profiles) / len(column_profiles)
        if total_missing > 50:
            issues.append("High missing data percentage across dataset")
            quality_score -= 30
        elif total_missing > 20:
            issues.append("Moderate missing data in some columns")
            quality_score -= 15
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > len(df) * 0.1:
            issues.append(f"High number of duplicate rows: {duplicate_count}")
            quality_score -= 20
        
        # Check for constant/quasi-constant features
        constant_features = [col.name for col in column_profiles if col.unique_count <= 1]
        if constant_features:
            issues.append(f"Constant features detected: {constant_features}")
            quality_score -= 10
        
        # Check for high cardinality categorical features
        high_cardinality = [col.name for col in column_profiles 
                          if col.dtype == 'object' and col.cardinality_ratio > 0.8]
        if high_cardinality:
            issues.append(f"High cardinality categorical features: {high_cardinality}")
            quality_score -= 10
        
        # Determine quality level
        if quality_score >= 85:
            return DataQuality.EXCELLENT, issues
        elif quality_score >= 70:
            return DataQuality.GOOD, issues
        elif quality_score >= 50:
            return DataQuality.FAIR, issues
        else:
            return DataQuality.POOR, issues
    
    async def analyze_dataset(self, 
                            file_path: str, 
                            dataset_id: str,
                            filename: str) -> Tuple[DataProfile, ActionPlan]:
        """
        Analyze dataset and generate profile and action plan.
        
        Args:
            file_path: Path to dataset file
            dataset_id: Unique dataset identifier
            filename: Original filename
            
        Returns:
            Tuple of (DataProfile, ActionPlan)
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing dataset: {filename}")
            
            # Load dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Basic dataset info
            shape = df.shape
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Analyze each column
            column_profiles = []
            target_candidates = []
            
            for column in df.columns:
                series = df[column]
                
                # Basic statistics
                null_count = series.isnull().sum()
                null_percentage = (null_count / len(series)) * 100
                unique_count = series.nunique()
                cardinality_ratio = unique_count / len(series)
                
                # Suggest role
                suggested_role = self._suggest_column_role(column, series, shape)
                if suggested_role == 'target_candidate':
                    target_candidates.append(column)
                
                # PII detection
                potential_pii = self._detect_pii(series, column)
                
                # Detailed statistics
                statistics = self._get_column_statistics(series)
                
                # Sample values (non-PII)
                sample_values = []
                if not potential_pii:
                    sample_values = series.dropna().head(5).tolist()
                    # Convert to JSON serializable format
                    sample_values = [str(val) if not isinstance(val, (int, float, str, bool)) else val 
                                   for val in sample_values]
                
                column_profile = ColumnProfile(
                    name=column,
                    dtype=str(series.dtype),
                    null_count=int(null_count),
                    null_percentage=float(null_percentage),
                    unique_count=int(unique_count),
                    cardinality_ratio=float(cardinality_ratio),
                    suggested_role=suggested_role,
                    potential_pii=potential_pii,
                    statistics=statistics,
                    sample_values=sample_values
                )
                
                column_profiles.append(column_profile)
            
            # Detect task type
            task_type = self._detect_task_type(df, target_candidates)
            
            # Assess data quality
            data_quality, issues = self._assess_data_quality(df, column_profiles)
            
            # Create data profile
            data_profile = DataProfile(
                dataset_id=dataset_id,
                filename=filename,
                shape=shape,
                size_mb=size_mb,
                columns=column_profiles,
                suggested_task_type=task_type,
                candidate_targets=target_candidates,
                data_quality=data_quality,
                issues=issues,
                metadata={
                    'analysis_time': time.time() - start_time,
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                },
                created_at=time.time()
            )
            
            # Generate action plan
            action_plan = self._generate_action_plan(data_profile, df)
            
            logger.info(f"Dataset analysis completed in {time.time() - start_time:.2f}s")
            
            return data_profile, action_plan
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            raise
    
    def _generate_action_plan(self, profile: DataProfile, df: pd.DataFrame) -> ActionPlan:
        """Generate action plan based on data profile."""
        
        # Determine if dataset is small enough for quick training
        is_small = (profile.size_mb <= self.small_threshold_mb and 
                   profile.shape[0] <= self.small_threshold_rows)
        
        quick_train_recommended = (is_small and 
                                 profile.data_quality in [DataQuality.EXCELLENT, DataQuality.GOOD] and
                                 len(profile.candidate_targets) > 0)
        
        # Chunk and embed for large datasets or when no clear ML task
        chunk_embed_recommended = (not is_small or 
                                 profile.suggested_task_type == TaskType.UNKNOWN or
                                 profile.data_quality == DataQuality.POOR)
        
        # Preprocessing recommendations
        preprocessing_steps = []
        
        if any(col.null_percentage > 10 for col in profile.columns):
            preprocessing_steps.append("Handle missing values")
        
        if any(col.potential_pii for col in profile.columns):
            preprocessing_steps.append("Remove or anonymize PII columns")
        
        if any(col.cardinality_ratio > 0.8 and col.dtype == 'object' for col in profile.columns):
            preprocessing_steps.append("Encode high-cardinality categorical features")
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            preprocessing_steps.append(f"Remove {duplicate_count} duplicate rows")
        
        # Estimate training time
        if profile.shape[0] < 1000:
            training_time = "< 1 minute"
        elif profile.shape[0] < 10000:
            training_time = "1-5 minutes"
        elif profile.shape[0] < 100000:
            training_time = "5-30 minutes"
        else:
            training_time = "30+ minutes"
        
        # Memory requirements
        memory_mb = profile.metadata.get('memory_usage_mb', profile.size_mb * 2)
        if memory_mb < 100:
            memory_req = "Low (< 100 MB)"
        elif memory_mb < 1000:
            memory_req = "Medium (100 MB - 1 GB)"
        else:
            memory_req = "High (> 1 GB)"
        
        # Suggest appropriate models
        suggested_models = []
        if profile.suggested_task_type == TaskType.CLASSIFICATION:
            if is_small:
                suggested_models = ["LightGBM", "TabKANet", "Random Forest"]
            else:
                suggested_models = ["LightGBM", "Neural Network"]
        elif profile.suggested_task_type == TaskType.REGRESSION:
            if is_small:
                suggested_models = ["LightGBM", "TabKANet", "Linear Regression"]
            else:
                suggested_models = ["LightGBM", "Deep Learning"]
        else:
            suggested_models = ["LightGBM (AutoML)", "Clustering", "Anomaly Detection"]
        
        # Priority actions
        priority_actions = []
        
        if quick_train_recommended:
            priority_actions.append({
                "label": "Quick Train Baseline",
                "action": "quick_train",
                "endpoint": "/api/train/quick",
                "description": "Train a baseline model with LightGBM"
            })
        
        if chunk_embed_recommended:
            priority_actions.append({
                "label": "Create Embeddings",
                "action": "create_embeddings", 
                "endpoint": "/api/data/create-embeddings",
                "description": "Chunk and embed dataset for chat-based exploration"
            })
        
        if preprocessing_steps:
            priority_actions.append({
                "label": "Data Preprocessing",
                "action": "preprocess",
                "endpoint": "/api/data/preprocess",
                "description": "Clean and prepare data for training"
            })
        
        return ActionPlan(
            quick_train_recommended=quick_train_recommended,
            chunk_embed_recommended=chunk_embed_recommended,
            preprocessing_steps=preprocessing_steps,
            estimated_training_time=training_time,
            memory_requirements=memory_req,
            suggested_models=suggested_models,
            priority_actions=priority_actions
        )
    
    def profile_to_json(self, profile: DataProfile) -> str:
        """Convert DataProfile to JSON string."""
        return json.dumps(asdict(profile), indent=2, default=str)
    
    def action_plan_to_json(self, action_plan: ActionPlan) -> str:
        """Convert ActionPlan to JSON string.""" 
        return json.dumps(asdict(action_plan), indent=2, default=str)
