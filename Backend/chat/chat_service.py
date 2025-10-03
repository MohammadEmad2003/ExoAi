"""
ChatService orchestrates the bilingual RAG pipeline.
Handles language detection, retrieval, context building, and response generation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time

from .model_client_gemini import ModelClientGemini, ChatMessage, ChatResponse, Language, get_gemini_client
from .vector_store import VectorStore, Document, get_vector_store

logger = logging.getLogger(__name__)

@dataclass
class ChatRequest:
    """Chat request with user context."""
    user_id: str
    message: str
    language: Optional[str] = None
    session_id: Optional[str] = None
    context_filters: Optional[Dict[str, Any]] = None

@dataclass
class ChatContext:
    """Retrieved context for RAG."""
    documents: List[Document]
    summary: str
    total_tokens: int
    retrieval_time: float

class ChatService:
    """Main chat service orchestrating bilingual RAG pipeline."""
    
    def __init__(self,
                 model_client: Optional[ModelClientGemini] = None,
                 vector_store: Optional[VectorStore] = None,
                 max_context_tokens: int = 2000,
                 retrieval_k: int = 5,
                 enable_context_compression: bool = True):
        """
        Initialize chat service.
        
        Args:
            model_client: Gemini model client
            vector_store: Vector store for document retrieval
            max_context_tokens: Maximum tokens for context
            retrieval_k: Number of documents to retrieve
            enable_context_compression: Whether to compress retrieved context
        """
        self.model_client = model_client or get_gemini_client()
        self.vector_store = vector_store or get_vector_store()
        self.max_context_tokens = max_context_tokens
        self.retrieval_k = retrieval_k
        self.enable_context_compression = enable_context_compression
        
        # Chat history storage (in production, use Redis/database)
        self.chat_histories: Dict[str, List[ChatMessage]] = {}
        
        logger.info("Initialized ChatService with RAG pipeline")
    
    def _get_session_key(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Generate session key for chat history."""
        return f"{user_id}:{session_id or 'default'}"
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for multilingual text)."""
        return max(1, len(text) // 4)
    
    async def _retrieve_context(self, 
                               query: str, 
                               language: Language,
                               filters: Optional[Dict[str, Any]] = None) -> ChatContext:
        """
        Retrieve relevant context documents using RAG.
        
        Args:
            query: User query
            language: Query language
            filters: Metadata filters for retrieval
            
        Returns:
            ChatContext with retrieved documents and summary
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.model_client.embed(query)
            
            # Retrieve similar documents
            results = await self.vector_store.query(
                embedding=query_embedding,
                k=self.retrieval_k,
                filter_metadata=filters
            )
            
            documents = [doc for doc, score in results if score > 0.5]  # Filter by relevance threshold
            
            # Build context summary
            context_text = ""
            total_tokens = 0
            
            for doc in documents:
                doc_text = f"Source ({doc.source_type}): {doc.content[:500]}...\n\n"
                doc_tokens = self._estimate_tokens(doc_text)
                
                if total_tokens + doc_tokens > self.max_context_tokens:
                    break
                
                context_text += doc_text
                total_tokens += doc_tokens
            
            # Compress context if enabled and necessary
            if self.enable_context_compression and total_tokens > self.max_context_tokens * 0.8:
                context_text = await self._compress_context(context_text, language)
                total_tokens = self._estimate_tokens(context_text)
            
            retrieval_time = time.time() - start_time
            
            logger.debug(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
            
            return ChatContext(
                documents=documents,
                summary=context_text,
                total_tokens=total_tokens,
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ChatContext(
                documents=[],
                summary="",
                total_tokens=0,
                retrieval_time=time.time() - start_time
            )
    
    async def _compress_context(self, context: str, language: Language) -> str:
        """
        Compress context using summarization (optional enhancement).
        For now, implements simple truncation with sentence boundaries.
        """
        try:
            if language == Language.ARABIC:
                # Arabic sentence boundaries
                sentences = [s.strip() for s in context.split('。') if s.strip()]
                separator = '。 '
            else:
                # English sentence boundaries
                sentences = [s.strip() for s in context.split('.') if s.strip()]
                separator = '. '
            
            # Keep most relevant sentences (first half + last quarter)
            if len(sentences) > 10:
                keep_first = len(sentences) // 2
                keep_last = len(sentences) // 4
                compressed_sentences = sentences[:keep_first] + sentences[-keep_last:]
                return separator.join(compressed_sentences)
            
            return context
            
        except Exception as e:
            logger.error(f"Error compressing context: {e}")
            # Fallback to simple truncation
            return context[:self.max_context_tokens * 4]
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process chat request with bilingual RAG pipeline.
        
        Args:
            request: Chat request with user message and context
            
        Returns:
            ChatResponse with reply, citations, and suggested actions
        """
        try:
            # Detect or use specified language
            language = Language.ARABIC if request.language == 'ar' else Language.ENGLISH
            if request.language is None:
                language = self.model_client.detect_language(request.message)
            
            # Get chat history
            session_key = self._get_session_key(request.user_id, request.session_id)
            chat_history = self.chat_histories.get(session_key, [])
            
            # Add user message to history
            user_message = ChatMessage(role="user", content=request.message)
            
            # Retrieve relevant context
            context = await self._retrieve_context(
                query=request.message,
                language=language,
                filters=request.context_filters
            )
            
            # Generate response
            response = await self.model_client.generate(
                chat_history=chat_history,
                context=context.summary,
                language=language,
                user_message=request.message
            )
            
            # Update chat history
            chat_history.append(user_message)
            assistant_message = ChatMessage(role="assistant", content=response.reply)
            chat_history.append(assistant_message)
            
            # Keep only last 20 messages
            self.chat_histories[session_key] = chat_history[-20:]
            
            # Enhance response with context metadata
            enhanced_citations = []
            for i, doc in enumerate(context.documents):
                citation = {
                    "source_id": doc.id,
                    "snippet": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "source_type": doc.source_type,
                    "language": doc.language,
                    "metadata": doc.metadata
                }
                enhanced_citations.append(citation)
            
            # Add retrieval metadata to response
            response.citations = enhanced_citations
            
            logger.info(f"Processed chat request in {language.value} with {len(enhanced_citations)} citations")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            
            # Return error response in appropriate language
            error_msg = "عذرًا، حدث خطأ في معالجة رسالتك." if language == Language.ARABIC else "Sorry, an error occurred processing your message."
            
            return ChatResponse(
                reply=error_msg,
                citations=[],
                suggested_actions=[],
                language=language,
                confidence=0.0
            )
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store for RAG retrieval.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Success status
        """
        try:
            # Generate embeddings for documents without them
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = await self.model_client.embed(doc.content)
            
            # Add to vector store
            success = await self.vector_store.upsert(documents)
            
            if success:
                logger.info(f"Added {len(documents)} documents to vector store")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def search_documents(self, 
                              query: str, 
                              k: int = 10,
                              language: Optional[str] = None,
                              source_type: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Search documents directly (for debugging/admin).
        
        Args:
            query: Search query
            k: Number of results
            language: Filter by language
            source_type: Filter by source type
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = await self.model_client.embed(query)
            
            # Build filters
            filters = {}
            if language:
                filters['language'] = language
            if source_type:
                filters['source_type'] = source_type
            
            # Search
            results = await self.vector_store.query(
                embedding=query_embedding,
                k=k,
                filter_metadata=filters if filters else None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_chat_history(self, user_id: str, session_id: Optional[str] = None) -> List[ChatMessage]:
        """Get chat history for a user session."""
        session_key = self._get_session_key(user_id, session_id)
        return self.chat_histories.get(session_key, [])
    
    def clear_chat_history(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """Clear chat history for a user session."""
        session_key = self._get_session_key(user_id, session_id)
        if session_key in self.chat_histories:
            del self.chat_histories[session_key]
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat service statistics."""
        return {
            "active_sessions": len(self.chat_histories),
            "total_messages": sum(len(history) for history in self.chat_histories.values()),
            "vector_store_stats": self.vector_store.get_stats(),
            "model_client_status": self.model_client.health_check()
        }

# Singleton instance
_chat_service_instance = None

def get_chat_service() -> ChatService:
    """Get singleton chat service instance."""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
