"""
Chunking, summarization, and embedding pipeline for large datasets.
Creates searchable embeddings of dataset segments for RAG chatbot.
"""

import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass
import asyncio
import time

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..chat.model_client_gemini import get_gemini_client
from ..chat.vector_store import Document, get_vector_store

logger = logging.getLogger(__name__)

@dataclass
class DatasetChunk:
    """A chunk of the dataset with metadata."""
    chunk_id: str
    dataset_id: str
    data: pd.DataFrame
    chunk_type: str  # stratified, clustered, random, temporal
    metadata: Dict[str, Any]
    summary: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class DatasetChunker:
    """Chunks large datasets into manageable pieces for embedding."""
    
    def __init__(self,
                 chunk_size: int = 1000,
                 max_chunks: int = 100,
                 clustering_features: int = 10):
        """
        Initialize dataset chunker.
        
        Args:
            chunk_size: Target size for each chunk
            max_chunks: Maximum number of chunks to create
            clustering_features: Max features to use for clustering
        """
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.clustering_features = clustering_features
        
        logger.info(f"Initialized DatasetChunker (chunk_size={chunk_size})")
    
    def _stratified_chunk(self, 
                         df: pd.DataFrame, 
                         target_column: str, 
                         dataset_id: str) -> List[DatasetChunk]:
        """Create stratified chunks based on target variable."""
        chunks = []
        
        try:
            # Get class distribution
            target_counts = df[target_column].value_counts()
            
            # Calculate proportional chunk sizes
            for class_label in target_counts.index:
                class_data = df[df[target_column] == class_label]
                class_size = len(class_data)
                
                # Number of chunks for this class
                num_chunks = max(1, class_size // self.chunk_size)
                num_chunks = min(num_chunks, self.max_chunks // len(target_counts))
                
                if num_chunks == 0:
                    continue
                
                # Split class data into chunks
                chunk_size_adjusted = class_size // num_chunks
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size_adjusted
                    end_idx = (i + 1) * chunk_size_adjusted if i < num_chunks - 1 else class_size
                    
                    chunk_data = class_data.iloc[start_idx:end_idx]
                    
                    chunk = DatasetChunk(
                        chunk_id=f"{dataset_id}_stratified_{class_label}_{i}",
                        dataset_id=dataset_id,
                        data=chunk_data,
                        chunk_type="stratified",
                        metadata={
                            "target_class": str(class_label),
                            "class_count": len(chunk_data),
                            "chunk_index": i,
                            "total_class_chunks": num_chunks
                        }
                    )
                    chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} stratified chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating stratified chunks: {e}")
            return []
    
    def _clustered_chunk(self, 
                        df: pd.DataFrame, 
                        dataset_id: str) -> List[DatasetChunk]:
        """Create chunks based on K-means clustering of numeric features."""
        chunks = []
        
        try:
            # Select numeric features for clustering
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                logger.warning("No numeric features for clustering, falling back to random chunking")
                return self._random_chunk(df, dataset_id)
            
            # Limit features and handle missing values
            if len(numeric_df.columns) > self.clustering_features:
                # Use PCA to reduce dimensionality
                pca = PCA(n_components=self.clustering_features)
                numeric_data = pca.fit_transform(numeric_df.fillna(numeric_df.mean()))
            else:
                numeric_data = numeric_df.fillna(numeric_df.mean()).values
            
            # Determine number of clusters
            n_samples = len(df)
            n_clusters = min(self.max_chunks, max(2, n_samples // self.chunk_size))
            
            # Perform clustering
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Create chunks from clusters
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = df[cluster_mask]
                
                if len(cluster_data) == 0:
                    continue
                
                # Calculate cluster center characteristics
                cluster_center = kmeans.cluster_centers_[cluster_id]
                center_description = self._describe_cluster_center(
                    cluster_center, numeric_df.columns[:len(cluster_center)]
                )
                
                chunk = DatasetChunk(
                    chunk_id=f"{dataset_id}_clustered_{cluster_id}",
                    dataset_id=dataset_id,
                    data=cluster_data,
                    chunk_type="clustered",
                    metadata={
                        "cluster_id": cluster_id,
                        "cluster_size": len(cluster_data),
                        "cluster_center_description": center_description,
                        "total_clusters": n_clusters
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} clustered chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating clustered chunks: {e}")
            return self._random_chunk(df, dataset_id)
    
    def _random_chunk(self, 
                     df: pd.DataFrame, 
                     dataset_id: str) -> List[DatasetChunk]:
        """Create random chunks as fallback method."""
        chunks = []
        
        try:
            n_samples = len(df)
            n_chunks = min(self.max_chunks, max(1, n_samples // self.chunk_size))
            
            # Shuffle and split
            shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            chunk_size_adjusted = n_samples // n_chunks
            
            for i in range(n_chunks):
                start_idx = i * chunk_size_adjusted
                end_idx = (i + 1) * chunk_size_adjusted if i < n_chunks - 1 else n_samples
                
                chunk_data = shuffled_df.iloc[start_idx:end_idx]
                
                chunk = DatasetChunk(
                    chunk_id=f"{dataset_id}_random_{i}",
                    dataset_id=dataset_id,
                    data=chunk_data,
                    chunk_type="random",
                    metadata={
                        "chunk_index": i,
                        "chunk_size": len(chunk_data),
                        "total_chunks": n_chunks
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} random chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating random chunks: {e}")
            return []
    
    def _temporal_chunk(self, 
                       df: pd.DataFrame, 
                       time_column: str, 
                       dataset_id: str) -> List[DatasetChunk]:
        """Create chunks based on temporal ordering."""
        chunks = []
        
        try:
            # Sort by time column
            sorted_df = df.sort_values(time_column)
            n_samples = len(sorted_df)
            n_chunks = min(self.max_chunks, max(1, n_samples // self.chunk_size))
            
            chunk_size_adjusted = n_samples // n_chunks
            
            for i in range(n_chunks):
                start_idx = i * chunk_size_adjusted
                end_idx = (i + 1) * chunk_size_adjusted if i < n_chunks - 1 else n_samples
                
                chunk_data = sorted_df.iloc[start_idx:end_idx]
                
                # Get time range
                time_start = chunk_data[time_column].min()
                time_end = chunk_data[time_column].max()
                
                chunk = DatasetChunk(
                    chunk_id=f"{dataset_id}_temporal_{i}",
                    dataset_id=dataset_id,
                    data=chunk_data,
                    chunk_type="temporal",
                    metadata={
                        "chunk_index": i,
                        "time_start": str(time_start),
                        "time_end": str(time_end),
                        "time_column": time_column,
                        "total_chunks": n_chunks
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} temporal chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating temporal chunks: {e}")
            return self._random_chunk(df, dataset_id)
    
    def _describe_cluster_center(self, 
                                center: np.ndarray, 
                                feature_names: List[str]) -> str:
        """Generate description of cluster center characteristics."""
        descriptions = []
        
        for i, (value, feature) in enumerate(zip(center, feature_names)):
            if i >= 5:  # Limit to top 5 features
                break
            descriptions.append(f"{feature}: {value:.2f}")
        
        return ", ".join(descriptions)
    
    def chunk_dataset(self,
                     df: pd.DataFrame,
                     dataset_id: str,
                     target_column: Optional[str] = None,
                     time_column: Optional[str] = None) -> List[DatasetChunk]:
        """
        Chunk dataset using appropriate strategy.
        
        Args:
            df: Dataset to chunk
            dataset_id: Unique dataset identifier
            target_column: Target column for stratified chunking
            time_column: Time column for temporal chunking
            
        Returns:
            List of dataset chunks
        """
        logger.info(f"Chunking dataset {dataset_id} with {len(df)} rows")
        
        # Choose chunking strategy
        if target_column and target_column in df.columns:
            # Use stratified chunking for supervised learning
            unique_targets = df[target_column].nunique()
            if unique_targets <= 100:  # Reasonable number of classes
                return self._stratified_chunk(df, target_column, dataset_id)
        
        if time_column and time_column in df.columns:
            # Use temporal chunking for time series data
            return self._temporal_chunk(df, time_column, dataset_id)
        
        # Check if clustering makes sense (sufficient numeric features)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 3:  # Minimum features for meaningful clustering
            return self._clustered_chunk(df, dataset_id)
        
        # Fallback to random chunking
        return self._random_chunk(df, dataset_id)

class DatasetSummarizer:
    """Generates human-readable summaries of dataset chunks."""
    
    def __init__(self, use_llm_summary: bool = True):
        """
        Initialize summarizer.
        
        Args:
            use_llm_summary: Whether to use LLM for enhanced summaries
        """
        self.use_llm_summary = use_llm_summary
        self.model_client = get_gemini_client() if use_llm_summary else None
        
        logger.info(f"Initialized DatasetSummarizer (use_llm={use_llm_summary})")
    
    def _basic_summary(self, chunk: DatasetChunk) -> str:
        """Generate basic statistical summary without LLM."""
        df = chunk.data
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Data chunk with {len(df)} rows and {len(df.columns)} columns")
        
        # Chunk type specific info
        if chunk.chunk_type == "stratified":
            target_class = chunk.metadata.get("target_class")
            summary_parts.append(f"Contains {target_class} class samples")
        elif chunk.chunk_type == "clustered":
            cluster_desc = chunk.metadata.get("cluster_center_description", "")
            summary_parts.append(f"Clustered data with characteristics: {cluster_desc}")
        elif chunk.chunk_type == "temporal":
            time_start = chunk.metadata.get("time_start")
            time_end = chunk.metadata.get("time_end")
            summary_parts.append(f"Temporal data from {time_start} to {time_end}")
        
        # Data characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            summary_parts.append(f"Numeric features: {', '.join(numeric_cols[:5])}")
            if len(numeric_cols) > 5:
                summary_parts.append(f"... and {len(numeric_cols) - 5} more")
        
        if len(categorical_cols) > 0:
            summary_parts.append(f"Categorical features: {', '.join(categorical_cols[:3])}")
            if len(categorical_cols) > 3:
                summary_parts.append(f"... and {len(categorical_cols) - 3} more")
        
        # Missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 5:
            summary_parts.append(f"Contains {missing_percentage:.1f}% missing values")
        
        return ". ".join(summary_parts) + "."
    
    async def _llm_enhanced_summary(self, chunk: DatasetChunk, basic_summary: str) -> str:
        """Enhance summary using LLM for better readability."""
        try:
            df = chunk.data
            
            # Prepare sample data for LLM context
            sample_data = df.head(3).to_string(index=False)
            
            prompt = f"""Analyze this dataset chunk and create a concise, informative summary for a data scientist.
            
Basic info: {basic_summary}

Sample data:
{sample_data}

Chunk metadata: {json.dumps(chunk.metadata, indent=2)}

Create a 2-3 sentence summary that highlights:
1. What this data represents
2. Key patterns or characteristics
3. Potential use cases or insights

Keep technical terms but make it accessible. Focus on actionable insights."""
            
            # Generate summary (simplified call)
            summary_response = await self.model_client.generate(
                chat_history=[],
                context="",
                user_message=prompt
            )
            
            return summary_response.reply
            
        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}")
            return basic_summary
    
    async def summarize_chunk(self, chunk: DatasetChunk) -> str:
        """
        Generate summary for a dataset chunk.
        
        Args:
            chunk: Dataset chunk to summarize
            
        Returns:
            Human-readable summary string
        """
        try:
            # Generate basic summary
            basic_summary = self._basic_summary(chunk)
            
            # Enhance with LLM if enabled
            if self.use_llm_summary and self.model_client:
                enhanced_summary = await self._llm_enhanced_summary(chunk, basic_summary)
                return enhanced_summary
            
            return basic_summary
            
        except Exception as e:
            logger.error(f"Error summarizing chunk {chunk.chunk_id}: {e}")
            return f"Dataset chunk with {len(chunk.data)} rows and {len(chunk.data.columns)} columns"

class ChunkEmbeddingPipeline:
    """Complete pipeline for chunking, summarizing, and embedding datasets."""
    
    def __init__(self,
                 chunk_size: int = 1000,
                 max_chunks: int = 100,
                 use_llm_summary: bool = True):
        """
        Initialize embedding pipeline.
        
        Args:
            chunk_size: Target size for chunks
            max_chunks: Maximum number of chunks
            use_llm_summary: Whether to use LLM for summaries
        """
        self.chunker = DatasetChunker(chunk_size, max_chunks)
        self.summarizer = DatasetSummarizer(use_llm_summary)
        self.model_client = get_gemini_client()
        self.vector_store = get_vector_store()
        
        logger.info("Initialized ChunkEmbeddingPipeline")
    
    async def process_dataset(self,
                            df: pd.DataFrame,
                            dataset_id: str,
                            filename: str,
                            target_column: Optional[str] = None,
                            time_column: Optional[str] = None,
                            language: str = "en") -> Dict[str, Any]:
        """
        Complete processing pipeline: chunk -> summarize -> embed -> store.
        
        Args:
            df: Dataset to process
            dataset_id: Unique dataset identifier
            filename: Original filename
            target_column: Target column name
            time_column: Time column name
            language: Dataset language
            
        Returns:
            Processing results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing dataset {dataset_id} with {len(df)} rows")
            
            # Step 1: Chunk dataset
            chunks = self.chunker.chunk_dataset(
                df=df,
                dataset_id=dataset_id,
                target_column=target_column,
                time_column=time_column
            )
            
            if not chunks:
                raise ValueError("No chunks were created from the dataset")
            
            # Step 2: Summarize and embed chunks
            documents = []
            processing_stats = {
                "chunks_created": len(chunks),
                "chunks_processed": 0,
                "chunks_failed": 0,
                "embedding_errors": []
            }
            
            for chunk in chunks:
                try:
                    # Generate summary
                    summary = await self.summarizer.summarize_chunk(chunk)
                    chunk.summary = summary
                    
                    # Generate embedding for summary
                    embedding = await self.model_client.embed(summary)
                    chunk.embedding = embedding
                    
                    # Create document for vector store
                    document = Document(
                        id=chunk.chunk_id,
                        content=summary,
                        embedding=embedding,
                        metadata={
                            "dataset_id": dataset_id,
                            "filename": filename,
                            "chunk_type": chunk.chunk_type,
                            "chunk_size": len(chunk.data),
                            "target_column": target_column,
                            "time_column": time_column,
                            **chunk.metadata
                        },
                        source_type="dataset",
                        language=language
                    )
                    
                    documents.append(document)
                    processing_stats["chunks_processed"] += 1
                    
                    logger.debug(f"Processed chunk {chunk.chunk_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
                    processing_stats["chunks_failed"] += 1
                    processing_stats["embedding_errors"].append(str(e))
            
            # Step 3: Store embeddings in vector store
            if documents:
                success = await self.vector_store.upsert(documents)
                if not success:
                    logger.error("Failed to store embeddings in vector store")
            
            processing_time = time.time() - start_time
            
            result = {
                "dataset_id": dataset_id,
                "processing_time": processing_time,
                "statistics": processing_stats,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                        "size": len(chunk.data),
                        "summary": chunk.summary,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks if chunk.summary
                ]
            }
            
            logger.info(f"Dataset processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in dataset processing pipeline: {e}")
            raise

# Factory function
def create_embedding_pipeline(**kwargs) -> ChunkEmbeddingPipeline:
    """Create embedding pipeline with custom parameters."""
    return ChunkEmbeddingPipeline(**kwargs)
