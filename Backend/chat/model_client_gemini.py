"""
Google Gemini model client for bilingual RAG chatbot.
Handles generation and embeddings with proper security and rate limiting.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import backoff

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import sentence_transformers
import numpy as np

logger = logging.getLogger(__name__)

class Language(Enum):
    ENGLISH = "en"
    ARABIC = "ar"

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ChatResponse:
    reply: str
    citations: List[Dict[str, str]]
    suggested_actions: List[Dict[str, str]]
    language: Language
    confidence: float = 1.0

class ModelClientGemini:
    """Google Gemini client with bilingual support and embeddings."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 max_retries: int = 3,
                 rate_limit_requests_per_minute: int = 60):
        """
        Initialize Gemini client with security best practices.
        
        Args:
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use
            embedding_model: Sentence transformer model for embeddings
            max_retries: Maximum retry attempts for failed requests
            rate_limit_requests_per_minute: Rate limiting for API calls
        """
        # Secure API key handling
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it with your Google AI Studio API key."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Rate limiting
        self.max_retries = max_retries
        self.rate_limit = rate_limit_requests_per_minute
        self.request_times = []
        
        # Initialize embedding model
        self.embedding_model = sentence_transformers.SentenceTransformer(embedding_model)
        
        # Safety settings for appropriate content
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    def _check_rate_limit(self):
        """Enforce rate limiting for API requests."""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def detect_language(self, text: str) -> Language:
        """
        Detect language of input text.
        Simple heuristic: check for Arabic characters.
        """
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return Language.ENGLISH
        
        arabic_ratio = arabic_chars / total_chars
        return Language.ARABIC if arabic_ratio > 0.3 else Language.ENGLISH
    
    def _build_system_prompt(self, language: Language, context: str = "") -> str:
        """Build system prompt based on language and context."""
        
        if language == Language.ARABIC:
            base_prompt = """أنت مساعد ذكي متخصص في تحليل البيانات والتعلم الآلي لمشروع Cosmic Analysts ExoAI. 
مهمتك هي مساعدة المستخدمين في:
- تحليل مجموعات البيانات الفلكية
- اقتراح نماذج التعلم الآلي المناسبة 
- شرح النتائج العلمية
- توجيه المستخدمين خلال عملية التدريب

قواعد مهمة:
1. احتفظ بأسماء النماذج باللغة الإنجليزية: TabKANet, QSVC, LightGBM
2. احتفظ بأسماء الملفات وصيغها باللغة الإنجليزية: .csv, .json, .pt
3. قدم إجابات واضحة ومفيدة مع اقتراحات عملية
4. اذكر مصادر المعلومات عند الإمكان"""
        else:
            base_prompt = """You are an intelligent assistant specialized in data analysis and machine learning for the Cosmic Analysts ExoAI project.
Your role is to help users with:
- Analyzing astronomical datasets
- Suggesting appropriate machine learning models
- Explaining scientific results  
- Guiding users through training processes

Important rules:
1. Keep model names in English: TabKANet, QSVC, LightGBM
2. Keep file names and formats in English: .csv, .json, .pt
3. Provide clear, helpful answers with actionable suggestions
4. Cite sources when possible"""
        
        if context:
            context_intro = "السياق المتاح:" if language == Language.ARABIC else "Available context:"
            base_prompt += f"\n\n{context_intro}\n{context}"
        
        return base_prompt
    
    def _build_chat_history_prompt(self, chat_history: List[ChatMessage], language: Language) -> str:
        """Convert chat history to prompt format."""
        if not chat_history:
            return ""
        
        history_lines = []
        for msg in chat_history[-10:]:  # Keep last 10 messages for context
            role = "المستخدم" if language == Language.ARABIC and msg.role == "user" else msg.role
            role = "المساعد" if language == Language.ARABIC and msg.role == "assistant" else role
            history_lines.append(f"{role}: {msg.content}")
        
        history_header = "سجل المحادثة:" if language == Language.ARABIC else "Chat History:"
        return f"{history_header}\n" + "\n".join(history_lines) + "\n"
    
    @backoff.on_exception(backoff.expo,
                         (google_exceptions.ResourceExhausted, 
                          google_exceptions.ServiceUnavailable,
                          google_exceptions.DeadlineExceeded),
                         max_tries=3)
    async def generate(self, 
                      chat_history: List[ChatMessage],
                      context: str = "",
                      language: Optional[Language] = None,
                      user_message: str = "") -> ChatResponse:
        """
        Generate response using Gemini with RAG context.
        
        Args:
            chat_history: Previous conversation messages
            context: Retrieved context from RAG pipeline
            language: Target language (auto-detected if None)
            user_message: Current user message
            
        Returns:
            ChatResponse with reply, citations, and suggested actions
        """
        try:
            self._check_rate_limit()
            
            # Detect language from user message or chat history
            if language is None:
                text_to_analyze = user_message or (chat_history[-1].content if chat_history else "")
                language = self.detect_language(text_to_analyze)
            
            # Build complete prompt
            system_prompt = self._build_system_prompt(language, context)
            chat_prompt = self._build_chat_history_prompt(chat_history, language)
            
            # Current user message
            current_message = ""
            if user_message:
                user_label = "المستخدم" if language == Language.ARABIC else "User"
                current_message = f"{user_label}: {user_message}\n"
            
            assistant_label = "المساعد" if language == Language.ARABIC else "Assistant"
            full_prompt = f"{system_prompt}\n\n{chat_prompt}{current_message}{assistant_label}:"
            
            # Generate response
            logger.info(f"Generating response in {language.value}")
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    full_prompt,
                    safety_settings=self.safety_settings,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=1024,
                    )
                )
            )
            
            reply_text = response.text if response.text else "عذرًا، لا يمكنني تقديم رد في الوقت الحالي." if language == Language.ARABIC else "Sorry, I cannot provide a response at this time."
            
            # Extract citations and suggested actions (basic implementation)
            citations = self._extract_citations(context)
            suggested_actions = self._generate_suggested_actions(reply_text, language)
            
            return ChatResponse(
                reply=reply_text,
                citations=citations,
                suggested_actions=suggested_actions,
                language=language,
                confidence=0.9  # Could implement confidence scoring
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_msg = "عذرًا، حدث خطأ في معالجة طلبك." if language == Language.ARABIC else "Sorry, an error occurred processing your request."
            return ChatResponse(
                reply=error_msg,
                citations=[],
                suggested_actions=[],
                language=language or Language.ENGLISH,
                confidence=0.0
            )
    
    def _extract_citations(self, context: str) -> List[Dict[str, str]]:
        """Extract citations from retrieved context."""
        citations = []
        
        # Simple citation extraction - in practice, this would be more sophisticated
        if "dataset" in context.lower():
            citations.append({
                "source_id": "dataset_profile",
                "snippet": context[:200] + "..." if len(context) > 200 else context,
                "source_type": "dataset"
            })
        
        if "model" in context.lower():
            citations.append({
                "source_id": "model_documentation", 
                "snippet": context[:200] + "..." if len(context) > 200 else context,
                "source_type": "documentation"
            })
        
        return citations
    
    def _generate_suggested_actions(self, reply: str, language: Language) -> List[Dict[str, str]]:
        """Generate context-appropriate suggested actions."""
        actions = []
        
        # Analyze reply content to suggest actions
        reply_lower = reply.lower()
        
        if "train" in reply_lower or "تدريب" in reply_lower:
            action_label = "بدء التدريب" if language == Language.ARABIC else "Start Training"
            actions.append({
                "label": action_label,
                "action_endpoint": "/api/train",
                "type": "training"
            })
        
        if "upload" in reply_lower or "رفع" in reply_lower:
            action_label = "رفع البيانات" if language == Language.ARABIC else "Upload Data"
            actions.append({
                "label": action_label,
                "action_endpoint": "/api/data/upload",
                "type": "upload"
            })
        
        if "export" in reply_lower or "تصدير" in reply_lower:
            action_label = "تصدير النموذج" if language == Language.ARABIC else "Export Model"
            actions.append({
                "label": action_label,
                "action_endpoint": "/api/models/export",
                "type": "export"
            })
        
        return actions
    
    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embeddings for text using multilingual model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Use sentence transformers for multilingual embeddings
            embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.embedding_model.encode(text, convert_to_numpy=True)
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the client is properly configured and operational."""
        try:
            # Test API key validity
            test_model = genai.GenerativeModel(self.model_name)
            test_response = test_model.generate_content(
                "Test connection",
                generation_config=genai.types.GenerationConfig(max_output_tokens=10)
            )
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "embedding_model": self.embedding_model.model_name,
                "api_accessible": bool(test_response.text)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name
            }

# Singleton instance for application use
_client_instance = None

def get_gemini_client() -> ModelClientGemini:
    """Get singleton Gemini client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = ModelClientGemini()
    return _client_instance
