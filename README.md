# AI-Powered RFP Response Agent System

Multi-agent AI system for automating B2B manufacturing RFP responses.

## Architecture

### Tech Stack
- **LLM**: Google Gemini Pro
- **Agent Framework**: LangChain
- **Backend**: Flask REST API
- **Frontend**: Streamlit
- **Database**: SQLite
- **Data Processing**: Pandas

### Agents
1. **Main Agent** - Orchestrates workflow
2. **Sales Agent** - RFP discovery & summarization
3. **Technical Agent** - SKU matching
4. **Pricing Agent** - Cost estimation

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Start backend
python backend/app.py

# 4. Start frontend (new terminal)
streamlit run frontend/streamlit_app.py
```

## Usage

1. Open Streamlit UI at http://localhost:8501
2. Paste RFP text in the input area
3. Click "Start RFP Analysis"
4. View automated matching and pricing results

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/rfp/analyze` - Analyze RFP
- `GET /api/catalog` - Get SKU catalog
- `POST /api/match` - Match SKUs to requirements
- `POST /api/pricing` - Calculate pricing

## Development

- Backend runs on http://localhost:5000
- Frontend runs on http://localhost:8501
- Database initialized in-memory on startup