#!/usr/bin/env python3
"""
Startup script for Cosmic Analysts ExoAI API server.
Handles environment setup and service initialization.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Configure logging for the application."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cosmic_analysts_api.log')
        ]
    )

def check_environment():
    """Check required environment variables."""
    required_vars = ['GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your environment or create a .env file")
        print("See env.example for reference")
        sys.exit(1)
    
    print("✅ Environment variables configured")

def create_directories():
    """Create necessary directories."""
    directories = [
        'data/uploads',
        'data/training_results', 
        'data/faiss_index',
        'models/trained',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main startup function."""
    print("🚀 Starting Cosmic Analysts ExoAI API...")
    
    # Load environment variables from .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment from .env file")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check environment
    check_environment()
    
    # Create directories
    create_directories()
    
    # Import and start the API
    try:
        import uvicorn
        from app.api.main import app
        
        # Configuration
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 8000))
        reload = os.getenv('DEBUG', 'false').lower() == 'true'
        
        logger.info(f"Starting API server on {host}:{port}")
        
        # Start server
        uvicorn.run(
            "app.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
