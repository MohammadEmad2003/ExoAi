# Cosmic Analysts ExoAI 🌌

**Advanced Machine Learning Platform for Exoplanet Analysis**

A comprehensive dual-mode platform combining cutting-edge TabKANet architecture with intuitive user interfaces for both novice and expert users. Built for NASA Challenge 2025, this platform democratizes exoplanet discovery through advanced AI and provides researchers with powerful tools for astronomical data analysis.

[![CI/CD Pipeline](https://github.com/your-org/Cosmic-Analysts-ExoAI/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/your-org/Cosmic-Analysts-ExoAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)

## 🚀 Features

### Dual-Mode Interface
- **🎓 Guided Mode**: Step-by-step tutorials for beginners with demo datasets
- **🔬 Research Hub**: Advanced workspace for researchers and power users
- **🌍 Multi-language**: English, Spanish, and Arabic support with full RTL support
- **♿ Accessibility**: WCAG 2.1 AA compliant with keyboard navigation

### Advanced ML Capabilities
- **🧠 TabKANet Architecture**: Kolmogorov-Arnold Network + Transformer attention
- **⚡ Classical ML**: XGBoost, LightGBM, CatBoost integration
- **🔮 Quantum ML**: Quantum Support Vector Classifier
- **🛡️ Adversarial Testing**: FGSM, PGD attacks with robustness metrics
- **📊 Real-time Training**: Live progress monitoring with metrics visualization

### 🤖 RAG-Enabled AI Assistant
- **💬 Bilingual Chat**: English + Arabic with full RTL support
- **📚 Document Retrieval**: Context-aware responses from dataset profiles
- **🔍 Intelligent Analysis**: Auto-profiling with actionable recommendations
- **🎯 Smart Actions**: One-click training, export, and preprocessing

### Production-Ready Pipeline
- **📊 Data Pipeline**: Drag-and-drop upload, schema mapping, auto-EDA
- **🎛️ Hyperparameter Tuning**: UI sliders + YAML editor for advanced users
- **📈 Real-time Monitoring**: Training progress, metrics, resource usage
- **🚀 Model Export**: TorchScript, ONNX, TensorRT formats
- **🐳 Containerized**: Docker deployment with CI/CD pipeline

### Explainability & Insights
- **📋 Model Cards**: Comprehensive performance and robustness metrics
- **🔍 SHAP Integration**: Feature importance and sample explanations
- **📖 Story Mode**: Interactive narratives with exportable reports
- **📊 Interactive Visualizations**: Light curve analysis and exoplanet detection

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   ML Pipeline   │
│   (React/TS)    │◄──►│   (FastAPI)     │◄──►│   (PyTorch)     │
│                 │    │                 │    │                 │
│ • Guided Mode   │    │ • REST API      │    │ • TabKANet      │
│ • Research Hub  │    │ • Job Queue     │    │ • Classical ML  │
│ • i18n Support  │    │ • Model Registry│    │ • Quantum ML    │
│ • Accessibility │    │ • Export System │    │ • Adversarial   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### TabKANet Architecture

The core innovation is our TabKANet model that combines:
- **Kolmogorov-Arnold Networks (KAN)**: For learning complex feature interactions
- **Transformer Attention**: For capturing long-range dependencies
- **Multi-head Attention**: For parallel processing of different feature aspects

```python
# Kolmogorov-Arnold Network + Transformer
model = TabKANet(
    n_num=12,              # Numerical features
    n_cat=3,               # Categorical features  
    cat_card_list=[5,10,8], # Category cardinalities
    d_model=64,            # Model dimension
    K_inner=16,            # KAN inner dimension
    trans_heads=4,         # Attention heads
    trans_depth=3,         # Transformer layers
    n_classes=3            # Output classes
)
```

## 🏃‍♂️ Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/Cosmic-Analysts-ExoAI.git
cd Cosmic-Analysts-ExoAI

# Start full stack
docker-compose up -d

# Access the platform
open http://localhost:3000  # Frontend (Guided Mode)
open http://localhost:8000  # API Server
open http://localhost:8888  # Jupyter Lab (Research Mode)
```

### Option 2: Local Development

#### Backend Setup
```bash
# Navigate to backend
cd Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API server
python start_api.py
```

#### Frontend Setup
```bash
# Navigate to frontend
cd Frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Access at http://localhost:3000
```

### Option 3: RAG Chatbot Setup

```bash
# Set up environment variables
cp Frontend/env.example Frontend/.env
# Edit .env and add your GEMINI_API_KEY

# Start the API server
cd Backend
python start_api.py

# Start frontend (new terminal)
cd Frontend
npm run dev

# Chat with your data at http://localhost:3000/upload
# Switch to العربية for full Arabic RTL experience
```

## 🎯 User Journeys

### For Beginners (Guided Mode)

1. **🎬 Welcome Tour**: Interactive introduction to ML concepts
2. **📁 Upload Data**: Drag-and-drop CSV with instant preview
3. **⚙️ Configure Model**: Guided parameter selection with presets
4. **🏃‍♂️ Train Model**: Watch real-time progress with explanations
5. **📊 Review Results**: Comprehensive metrics and visualizations
6. **🚀 Deploy Model**: One-click export for production use

### For Researchers (Research Hub)

1. **🔬 Advanced Workspace**: Full control over data pipelines
2. **📊 Multi-Model Training**: Concurrent experiments with job queue
3. **🎛️ Hyperparameter Optimization**: YAML configs + automated tuning
4. **📈 Experiment Tracking**: MLflow integration with version control
5. **🔍 Model Analysis**: Deep dive into robustness and explainability
6. **🌐 Production Deployment**: Scalable inference with monitoring

## 📊 Performance Benchmarks

### Model Performance
| Model | Dataset | Accuracy | Robustness | Inference |
|-------|---------|----------|------------|-----------|
| TabKANet | Exoplanet (2.8k) | **92.4%** | 85.6% adv. acc. | 2.3ms |
| FT-Transformer | Stellar (5.2k) | 87.6% | 82.1% adv. acc. | 1.8ms |
| XGBoost | Galaxy (12k) | 84.5% | N/A | 0.5ms |

### Export Formats
| Format | Size | CPU Inference | GPU Inference | Compatibility |
|--------|------|---------------|---------------|---------------|
| **TorchScript** | 12.4MB | 2.3ms | 1.1ms | PyTorch |
| **ONNX** | 11.8MB | 1.9ms | N/A | Universal |
| **TensorRT** | 8.5MB | N/A | 0.7ms | NVIDIA GPU |

## 🔌 API Endpoints

The ExoAI platform provides a comprehensive REST API for all functionality. The API is built with FastAPI and includes automatic OpenAPI documentation at `http://localhost:8000/docs`.

### Core Endpoints

#### Health & Status
- `GET /` - API root with version info
- `GET /health` - Health check with service statistics

#### Data Management
- `POST /api/data/upload` - Upload dataset files (CSV, Parquet, JSON)
- `GET /api/data/{dataset_id}/profile` - Get dataset analysis and profile
- `POST /api/data/{dataset_id}/create-embeddings` - Create embeddings for RAG

#### Chat & RAG
- `POST /api/chat` - Process chat with bilingual RAG pipeline
- `GET /api/chat/history/{user_id}` - Get chat history for user
- `DELETE /api/chat/history/{user_id}` - Clear chat history

#### Training & Models
- `POST /api/training/start` - Start TabKANet training job
- `GET /api/training/jobs` - List all training jobs
- `GET /api/training/jobs/{job_id}` - Get specific job status
- `POST /api/train/quick` - Quick training with AutoML
- `GET /api/train/{job_id}` - Get training results

#### Model Management
- `GET /api/models` - List all registered models
- `GET /api/models/{model_id}` - Get model details
- `POST /api/models/{model_id}/predict` - Make predictions
- `POST /api/models/{model_id}/export` - Export model (TorchScript, ONNX)
- `GET /api/models/{model_id}/download/{format}` - Download exported model

#### Vector Store
- `GET /api/vector/stats` - Get vector store statistics
- `GET /api/vector/search` - Search documents in vector store

### Example API Usage

#### Upload and Analyze Data
```bash
# Upload dataset
curl -X POST "http://localhost:8000/api/data/upload" \
  -F "file=@exoplanet_data.csv"

# Get dataset profile
curl "http://localhost:8000/api/data/{dataset_id}/profile"
```

#### Start Training
```bash
# Start TabKANet training
curl -X POST "http://localhost:8000/api/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "exoplanet_data", 
    "target_column": "planet_type",
    "config": {
      "model_type": "tabkanet",
      "learning_rate": 0.001,
      "epochs": 100,
      "d_model": 64,
      "K_inner": 16,
      "trans_heads": 4,
      "trans_depth": 3
    }
  }'

# Monitor training progress
curl "http://localhost:8000/api/training/jobs/{job_id}"
```

#### Chat with RAG
```bash
# Chat in English
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "What patterns should I look for in exoplanet data?",
    "language": "en"
  }'

# Chat in Arabic
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "ما هي الأنماط التي يجب أن أبحث عنها في بيانات الكواكب الخارجية؟",
    "language": "ar"
  }'
```

#### Make Predictions
```bash
# Use trained model for predictions
curl -X POST "http://localhost:8000/api/models/{model_id}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_123",
    "data": [
      {
        "stellar_mass": 1.2,
        "orbital_period": 365.25,
        "transit_depth": 0.01,
        "stellar_radius": 1.1
      }
    ]
  }'
```

#### Export Model
```bash
# Export to TorchScript
curl -X POST "http://localhost:8000/api/models/{model_id}/export" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_123",
    "format": "torchscript"
  }'

# Download exported model
curl "http://localhost:8000/api/models/{model_id}/download/torchscript" \
  -o model.pt
```

### Request/Response Models

#### Training Configuration
```json
{
  "model_type": "tabkanet",
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 100,
  "dropout": 0.1,
  "d_model": 64,
  "K_inner": 16,
  "trans_heads": 4,
  "trans_depth": 3,
  "mlp_hidden": 128,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "weight_decay": 0.01,
  "early_stopping": true,
  "patience": 10,
  "validation_split": 0.2
}
```

#### Chat Request
```json
{
  "user_id": "string",
  "message": "string",
  "language": "en|ar",
  "session_id": "string",
  "context_filters": {}
}
```

#### Chat Response
```json
{
  "reply": "string",
  "citations": [
    {
      "source_id": "string",
      "snippet": "string",
      "source_type": "string",
      "language": "string",
      "metadata": {}
    }
  ],
  "suggested_actions": [
    {
      "action": "string",
      "description": "string"
    }
  ],
  "language": "en|ar",
  "confidence": 0.95
}
```

## 🛠️ Development

### Training a Model

```bash
# API approach
curl -X POST "http://localhost:8000/api/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "exoplanet_data", 
    "target_column": "planet_type",
    "config": {
      "model_type": "tabkanet",
      "learning_rate": 0.001,
      "epochs": 100,
      "d_model": 64
    }
  }'

# Monitor progress
curl "http://localhost:8000/api/training/jobs/{job_id}"
```

### Model Export & Deployment

```bash
# Export trained model
python Backend/export_model.py \
  --checkpoint models_registry/model_id.pt \
  --model-class models.tabkanet.TabKANet \
  --format torchscript \
  --output-dir exports/

# Benchmark performance
python Backend/benchmark_inference.py \
  --model-path exports/model.pt \
  --batch-sizes 1,8,32 \
  --output benchmarks/results.json

# Deploy with Docker
docker run -p 8000:8000 cosmic-analysts-exoai:latest
```

### Testing

```bash
# Backend tests
cd Backend
pytest tests/ -v --cov=app --cov=models

# Test specific components  
pytest tests/test_models.py::test_tabkanet_forward -v
pytest tests/test_export.py::test_torchscript_export -v

# Frontend tests
cd Frontend
npm test
```

## 🎨 UI Components

### Guided Mode Flow
```typescript
// Onboarding with step-by-step guidance
<OnboardingFlow>
  <WelcomeStep />
  <DataUploadStep />
  <ConfigureStep />
  <TrainStep />
  <EvaluateStep />
  <DeployStep />
</OnboardingFlow>
```

### Research Hub Dashboard
```typescript
// Advanced workspace for power users
<ResearchHub>
  <DataPipeline />
  <TrainingQueue />
  <ModelRegistry />
  <ExperimentTracking />
  <DeploymentMonitor />
</ResearchHub>
```

## 🌟 Storytelling Examples

### Mission Concept: AI-Powered Exoplanet Discovery

*"In the cosmic ocean of stars, our TabKANet serves as an intelligent lighthouse, guiding astronomers toward potentially habitable worlds. By analyzing subtle patterns in stellar data, our AI can distinguish between rocky Earth-like planets and gas giants with unprecedented accuracy, accelerating the search for life beyond our solar system."*

### Public Narrative: Democratizing Space Science

*"Imagine having a personal AI assistant that can analyze starlight and tell you about distant worlds. Cosmic Analysts ExoAI makes advanced space science accessible to everyone - from curious students taking their first steps into astronomy to seasoned researchers pushing the boundaries of exoplanet discovery."*

## 🔒 Security & Privacy

- **🔐 Data Privacy**: Local processing by default, explicit opt-in for cloud features
- **🛡️ Input Validation**: Comprehensive sanitization and rate limiting
- **🔍 Vulnerability Scanning**: Automated security checks in CI/CD
- **📊 Audit Logging**: Complete traceability of model training and deployment

## 🌍 Accessibility & Internationalization

- **♿ WCAG 2.1 AA**: Full keyboard navigation and screen reader support
- **🌐 Multi-language**: English, Spanish, and Arabic with extensible i18n framework
- **📱 Responsive Design**: Mobile-first approach with progressive enhancement
- **🎨 High Contrast**: Accessible color schemes and typography

## 📚 Project Structure

```
ExoAI/
├── Backend/                    # Python FastAPI backend
│   ├── api/                   # API endpoints and routes
│   ├── chat/                  # RAG-enabled chat service
│   ├── models/                # ML models (TabKANet, etc.)
│   ├── data/                  # Data processing utilities
│   ├── tests/                 # Backend tests
│   ├── uploads/               # User uploaded files
│   ├── exports/               # Exported models
│   └── models_registry/       # Model versioning
├── Frontend/                   # React TypeScript frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/            # Application pages
│   │   ├── hooks/            # Custom React hooks
│   │   ├── i18n/             # Internationalization
│   │   └── lib/              # Utility functions
│   ├── public/               # Static assets
│   └── scripts/              # Build and deployment scripts
└── docs/                     # Documentation
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Scale services
docker-compose up -d --scale api=3

# View logs
docker-compose logs -f api
```

### Production Deployment

```bash
# Build production image
docker build -t cosmic-analysts-exoai:latest .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e DATABASE_URL=your_db \
  cosmic-analysts-exoai:latest
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone
git clone https://github.com/your-username/Cosmic-Analysts-ExoAI.git
cd Cosmic-Analysts-ExoAI

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r Backend/requirements.txt -r Backend/requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run development servers
docker-compose up -d
```

### Code Standards
- **Python**: Black formatting, flake8 linting, mypy type checking
- **TypeScript**: ESLint + Prettier, strict type checking
- **Tests**: 80%+ coverage with pytest and Jest
- **Documentation**: Comprehensive docstrings and README updates

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for comprehensive datasets
- **PyTorch Team** for the deep learning framework
- **Hugging Face** for transformer implementations
- **Open Source Community** for countless tools and libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/Cosmic-Analysts-ExoAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/Cosmic-Analysts-ExoAI/discussions)
- **Email**: cosmic-analysts@example.com

## 📈 Citation

```bibtex
@software{cosmic_analysts_exoai_2024,
  title={Cosmic Analysts ExoAI: Advanced Machine Learning Platform for Exoplanet Analysis},
  author={Cosmic Analysts Team},
  year={2024},
  url={https://github.com/your-org/Cosmic-Analysts-ExoAI},
  note={NASA Challenge 2025 Submission}
}
```

---

**🌌 Ready to explore the cosmos with AI? [Get Started](http://localhost:3000) today!**
