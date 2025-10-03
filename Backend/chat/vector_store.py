"""
Vector store wrapper supporting FAISS (local/dev) and Pinecone (production).
Handles document storage, retrieval, and similarity search for RAG pipeline.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

# FAISS for local development
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

# Pinecone for production
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not available. Install with: pip install pinecone-client")

logger = logging.getLogger(__name__)

class VectorStoreType(Enum):
    FAISS = "faiss"
    PINECONE = "pinecone"
    MILVUS = "milvus"

@dataclass
class Document:
    """Document with metadata for vector storage."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    source_type: str = "general"  # dataset, documentation, model_card, etc.
    language: str = "en"
    timestamp: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        doc_dict = asdict(self)
        if self.embedding is not None:
            doc_dict['embedding'] = self.embedding.tolist()
        return doc_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary."""
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)

class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def upsert(self, documents: List[Document]) -> bool:
        """Insert or update documents."""
        pass
    
    @abstractmethod
    async def query(self, 
                   embedding: np.ndarray, 
                   k: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Query for similar documents."""
        pass
    
    @abstractmethod
    async def delete(self, document_ids: List[str]) -> bool:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass

class FAISSVectorStore(VectorStoreBase):
    """FAISS-based vector store for local development."""
    
    def __init__(self, 
                 dimension: int = 768,
                 index_path: str = "data/faiss_index",
                 metadata_path: str = "data/faiss_metadata.json"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_path: Path to save FAISS index
            metadata_path: Path to save document metadata
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Create directories
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.metadata_store: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        
        # Load existing index and metadata
        self._load_index()
        
        logger.info(f"Initialized FAISS vector store with {self.index.ntotal} documents")
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata_json = json.load(f)
                    self.metadata_store = {
                        doc_id: Document.from_dict(doc_data)
                        for doc_id, doc_data in metadata_json.items()
                    }
                    
                    # Rebuild ID to index mapping
                    self.id_to_index = {
                        doc_id: idx for idx, doc_id in enumerate(self.metadata_store.keys())
                    }
                    
                logger.info(f"Loaded {len(self.metadata_store)} document metadata entries")
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = {}
            self.id_to_index = {}
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            metadata_json = {
                doc_id: doc.to_dict() for doc_id, doc in self.metadata_store.items()
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_json, f, ensure_ascii=False, indent=2)
                
            logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    async def upsert(self, documents: List[Document]) -> bool:
        """Insert or update documents in FAISS index."""
        try:
            new_embeddings = []
            new_docs = []
            
            for doc in documents:
                if doc.embedding is None:
                    logger.warning(f"Document {doc.id} has no embedding, skipping")
                    continue
                
                # Check if document already exists
                if doc.id in self.metadata_store:
                    # Update existing document
                    old_index = self.id_to_index[doc.id]
                    # FAISS doesn't support direct updates, so we'll rebuild if needed
                    logger.warning(f"Document {doc.id} already exists. Consider rebuilding index for updates.")
                
                new_embeddings.append(doc.embedding)
                new_docs.append(doc)
                self.metadata_store[doc.id] = doc
            
            if new_embeddings:
                # Normalize embeddings for cosine similarity
                embeddings_array = np.array(new_embeddings).astype('float32')
                faiss.normalize_L2(embeddings_array)
                
                # Add to index
                start_index = self.index.ntotal
                self.index.add(embeddings_array)
                
                # Update ID mapping
                for i, doc in enumerate(new_docs):
                    self.id_to_index[doc.id] = start_index + i
                
                # Save to disk
                self._save_index()
                
                logger.info(f"Added {len(new_embeddings)} documents to FAISS index")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            return False
    
    async def query(self, 
                   embedding: np.ndarray, 
                   k: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Query FAISS index for similar documents."""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = embedding.copy().astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                
                # Find document by index
                doc_id = None
                for doc_id, doc_idx in self.id_to_index.items():
                    if doc_idx == idx:
                        break
                
                if doc_id and doc_id in self.metadata_store:
                    doc = self.metadata_store[doc_id]
                    
                    # Apply metadata filtering
                    if filter_metadata:
                        if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                            continue
                    
                    results.append((doc, float(score)))
            
            # Sort by score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error querying FAISS index: {e}")
            return []
    
    async def delete(self, document_ids: List[str]) -> bool:
        """Delete documents (requires index rebuild for FAISS)."""
        try:
            deleted_count = 0
            for doc_id in document_ids:
                if doc_id in self.metadata_store:
                    del self.metadata_store[doc_id]
                    if doc_id in self.id_to_index:
                        del self.id_to_index[doc_id]
                    deleted_count += 1
            
            if deleted_count > 0:
                # Rebuild index (FAISS doesn't support deletion)
                remaining_docs = list(self.metadata_store.values())
                embeddings = [doc.embedding for doc in remaining_docs if doc.embedding is not None]
                
                if embeddings:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    embeddings_array = np.array(embeddings).astype('float32')
                    faiss.normalize_L2(embeddings_array)
                    self.index.add(embeddings_array)
                    
                    # Rebuild ID mapping
                    self.id_to_index = {
                        doc.id: idx for idx, doc in enumerate(remaining_docs)
                        if doc.embedding is not None
                    }
                else:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.id_to_index = {}
                
                self._save_index()
                logger.info(f"Deleted {deleted_count} documents and rebuilt index")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS vector store statistics."""
        return {
            "type": "faiss",
            "total_documents": self.index.ntotal,
            "dimension": self.dimension,
            "index_path": self.index_path,
            "metadata_count": len(self.metadata_store)
        }

class PineconeVectorStore(VectorStoreBase):
    """Pinecone-based vector store for production."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 environment: str = "us-west1-gcp",
                 index_name: str = "cosmic-analysts-rag",
                 dimension: int = 768):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key (from env if None)
            environment: Pinecone environment
            index_name: Name of Pinecone index
            dimension: Embedding dimension
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=environment)
        
        # Create or connect to index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            logger.info(f"Created Pinecone index: {index_name}")
        
        self.index = pinecone.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    async def upsert(self, documents: List[Document]) -> bool:
        """Upsert documents to Pinecone."""
        try:
            vectors = []
            for doc in documents:
                if doc.embedding is None:
                    continue
                
                vector_data = {
                    "id": doc.id,
                    "values": doc.embedding.tolist(),
                    "metadata": {
                        "content": doc.content,
                        "source_type": doc.source_type,
                        "language": doc.language,
                        **doc.metadata
                    }
                }
                vectors.append(vector_data)
            
            if vectors:
                # Upsert in batches of 100
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                
                logger.info(f"Upserted {len(vectors)} documents to Pinecone")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {e}")
            return False
    
    async def query(self,
                   embedding: np.ndarray,
                   k: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Query Pinecone for similar documents."""
        try:
            query_response = self.index.query(
                vector=embedding.tolist(),
                top_k=k,
                include_values=False,
                include_metadata=True,
                filter=filter_metadata
            )
            
            results = []
            for match in query_response.matches:
                metadata = match.metadata
                doc = Document(
                    id=match.id,
                    content=metadata.get('content', ''),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ['content', 'source_type', 'language']},
                    source_type=metadata.get('source_type', 'general'),
                    language=metadata.get('language', 'en')
                )
                results.append((doc, match.score))
            
            logger.debug(f"Found {len(results)} similar documents in Pinecone")
            return results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return []
    
    async def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone."""
        try:
            self.index.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "type": "pinecone",
                "total_documents": stats.total_vector_count,
                "dimension": self.dimension,
                "index_name": self.index_name,
                "environment": self.environment
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
            return {"type": "pinecone", "error": str(e)}

class VectorStore:
    """Unified vector store interface that auto-selects backend."""
    
    def __init__(self,
                 store_type: Optional[VectorStoreType] = None,
                 dimension: int = 768,
                 **kwargs):
        """
        Initialize vector store with auto-selection.
        
        Args:
            store_type: Specific store type (auto-selected if None)
            dimension: Embedding dimension
            **kwargs: Additional arguments for specific stores
        """
        self.dimension = dimension
        
        # Auto-select store type if not specified
        if store_type is None:
            if os.getenv('PINECONE_API_KEY'):
                store_type = VectorStoreType.PINECONE
            elif FAISS_AVAILABLE:
                store_type = VectorStoreType.FAISS
            else:
                raise RuntimeError("No vector store backend available")
        
        # Initialize appropriate backend
        if store_type == VectorStoreType.FAISS:
            self.backend = FAISSVectorStore(dimension=dimension, **kwargs)
        elif store_type == VectorStoreType.PINECONE:
            self.backend = PineconeVectorStore(dimension=dimension, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        self.store_type = store_type
        logger.info(f"Initialized vector store with {store_type.value} backend")
    
    async def upsert(self, documents: List[Document]) -> bool:
        """Insert or update documents."""
        return await self.backend.upsert(documents)
    
    async def query(self,
                   embedding: np.ndarray,
                   k: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Query for similar documents."""
        return await self.backend.query(embedding, k, filter_metadata)
    
    async def delete(self, document_ids: List[str]) -> bool:
        """Delete documents by ID."""
        return await self.backend.delete(document_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.backend.get_stats()

# Singleton instance
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get singleton vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
