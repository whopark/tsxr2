# TSXr2

**Gemini-Powered Chest X-ray Analysis & Reporting System**

A hybrid AI pipeline that combines local vision model inference (DenseNet121) with Google's Gemini API for comprehensive chest X-ray analysis and clinical report generation.

## Features

- **DICOM Processing** - Load, validate, and preprocess DICOM chest X-rays with HIPAA-compliant PHI removal
- **Multi-label Classification** - Detect 14 pathological findings using a DenseNet121-based model
- **Clinical Report Generation** - Generate structured radiology reports via Gemini API
- **Reliability Assessment** - Confidence scoring, quality validation, and automatic fallback
- **RESTful API** - FastAPI-based endpoints for PACS/EMR integration

## Detected Findings

The TSXr model detects the following 14 pathological findings:

| Finding | Finding | Finding |
|---------|---------|---------|
| Atelectasis | Cardiomegaly | Consolidation |
| Edema | Effusion | Emphysema |
| Fibrosis | Hernia | Infiltration |
| Mass | Nodule | Pleural Thickening |
| Pneumonia | Pneumothorax | |

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Google Gemini API key

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd tsxr2

# Install with development dependencies
pip install -e ".[dev]"

# Or using uv (recommended)
uv sync --all-extras
```

### Environment Setup

Create a `.env` file or set environment variables:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Quick Start

### Start the API server

```bash
# Development mode with auto-reload
uvicorn tsxr2.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn tsxr2.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access the API documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## API Endpoints

### Analysis Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/full-report` | POST | **Recommended** - Complete pipeline with reliability info |
| `/analyze` | POST | TSXr model inference only (returns TSXrOutput) |
| `/report` | POST | Gemini clinical report generation |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/preprocess` | POST | Preprocess DICOM and return base64 image |
| `/health` | GET | Simple health check |
| `/health/detailed` | GET | Component status (model, device) |
| `/metrics` | GET | Request counters and uptime |

### Example: Full Report

```bash
curl -X POST "http://localhost:8000/full-report" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.dcm"
```

Response:
```json
{
  "tsxr_output": {
    "metadata": {"model_version": "tsxr-v2.1", "timestamp": "..."},
    "findings": [{"label": "Cardiomegaly", "probability": 0.85, ...}],
    "global_scores": {"abnormality_score": 0.85, "confidence_index": 0.78}
  },
  "gemini_report": {
    "findings": "The chest X-ray shows evidence of cardiomegaly...",
    "impression": "Cardiomegaly with mild bilateral pleural effusions.",
    "recommendations": "Recommend echocardiogram..."
  },
  "reliability": {
    "confidence": {"level": "medium", "warnings": [...]},
    "quality": {"is_acceptable": true},
    "used_fallback": false
  }
}
```

## Architecture

```
DICOM Input
    │
    ▼
┌─────────────────┐
│  Preprocessing  │ ← PHI removal, normalization, 512x512 resize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   TSXr Model    │ ← DenseNet121, 14-class multi-label
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reliability     │ ← Confidence assessment, quality validation
│ Assessment      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini API     │ ← Clinical report generation (with fallback)
└────────┬────────┘
         │
         ▼
    Full Report
```

## Development

### Run tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=tsxr2

# Run specific test file
uv run pytest tests/test_api.py -v
```

### Code quality

```bash
# Run linter
uv run ruff check src tests

# Format code
uv run ruff format src tests
```

### Project Structure

```
tsxr2/
├── src/tsxr2/
│   ├── api/           # FastAPI application
│   │   └── main.py    # API endpoints
│   ├── gemini/        # Gemini integration
│   │   ├── client.py  # API client
│   │   ├── prompts.py # Prompt templates
│   │   └── parser.py  # Response parser
│   ├── preprocessing/ # DICOM processing
│   │   ├── dicom_loader.py
│   │   ├── phi_remover.py
│   │   └── normalizer.py
│   ├── reliability/   # Reliability layer
│   │   ├── confidence.py
│   │   ├── quality.py
│   │   ├── fallback.py
│   │   └── validation.py
│   ├── schemas/       # Pydantic models
│   │   ├── tsxr_output.py
│   │   ├── gemini_report.py
│   │   └── full_report.py
│   ├── tsxr/          # Vision model
│   │   ├── model_loader.py
│   │   ├── inference.py
│   │   └── formatter.py
│   └── logging/       # Audit logging
│       └── audit.py
├── tests/             # Test suite (65+ tests)
├── pyproject.toml     # Project configuration
└── README.md
```

## Deployment

### Docker (recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "tsxr2.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `CUDA_VISIBLE_DEVICES` | No | GPU device selection |

## License

MIT License

## Disclaimer

This software is for research and educational purposes only. It is not intended for clinical diagnosis. All AI-generated findings require verification by a qualified radiologist.
