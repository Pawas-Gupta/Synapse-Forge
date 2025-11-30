# RFP Agent System - Deployment Guide

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment tool
- Redis (optional, for production caching)
- 4GB RAM minimum (8GB recommended)
- 2GB disk space

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd rfp-agent-system

# Create virtual environment
python -m venv myenv

# Activate virtual environment
# Windows:
.\myenv\Scripts\activate
# Linux/Mac:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
# Minimum required:
# - GROQ_API_KEY (get from https://console.groq.com)
```

### 4. Initialize Database

```bash
# The database will be created automatically on first run
# Or manually initialize:
python -c "from backend.database import get_db; get_db()"
```

### 5. Build FAISS Index

```bash
# Build semantic search index
python -c "from backend.tools.sku_tools import SemanticMatcher; from backend.database import get_db; matcher = SemanticMatcher(get_db().get_connection()); matcher.build_index()"
```

### 6. Start Backend

```bash
python backend/app.py
```

Backend will be available at: http://127.0.0.1:5000

### 7. Start Frontend (Optional)

```bash
# In a new terminal
streamlit run frontend/streamlit_app.py
```

Frontend will be available at: http://localhost:8501

## Production Deployment

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t rfp-agent-system .

# Run container
docker run -d \
  -p 5000:5000 \
  -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  -e REDIS_URL=redis://redis:6379/0 \
  --name rfp-agent \
  rfp-agent-system
```

### Option 2: Systemd Service (Linux)

Create `/etc/systemd/system/rfp-agent.service`:

```ini
[Unit]
Description=RFP Agent System Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/rfp-agent-system
Environment="PATH=/opt/rfp-agent-system/myenv/bin"
ExecStart=/opt/rfp-agent-system/myenv/bin/python backend/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rfp-agent
sudo systemctl start rfp-agent
sudo systemctl status rfp-agent
```

### Option 3: Gunicorn (Production WSGI)

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

### Option 4: Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /metrics {
        proxy_pass http://127.0.0.1:5000/metrics;
        # Optional: Restrict access
        allow 10.0.0.0/8;
        deny all;
    }
}
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| GROQ_API_KEY | Groq API key for LLM | `gsk_...` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| REDIS_URL | Redis connection URL | `memory://` |
| DATABASE_URL | Database connection string | `sqlite:///data/rfp_system.db` |
| FLASK_HOST | Flask bind address | `127.0.0.1` |
| FLASK_PORT | Flask port | `5000` |
| ADMIN_API_TOKEN | Admin authentication token | None |
| USER_API_TOKENS | User tokens (comma-separated) | None |

See `.env.example` for complete list.

## Security Checklist

- [ ] Change default admin token
- [ ] Enable HTTPS in production
- [ ] Configure firewall rules
- [ ] Set up Redis authentication
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Use strong API tokens
- [ ] Enable authentication on all endpoints
- [ ] Set up monitoring and alerts
- [ ] Regular security updates

## Monitoring

### Health Check

```bash
curl http://localhost:5000/api/health
```

### Prometheus Metrics

```bash
curl http://localhost:5000/metrics
```

### Cache Metrics

```bash
curl http://localhost:5000/api/cache-metrics
```

### Database Pool Status

```bash
curl http://localhost:5000/api/pool-status
```

## Performance Tuning

### For High Traffic

1. **Enable Redis caching**:
   ```bash
   REDIS_URL=redis://localhost:6379/0
   ```

2. **Increase connection pool**:
   ```bash
   DB_POOL_SIZE=20
   DB_POOL_MAX_OVERFLOW=40
   ```

3. **Use multiple workers**:
   ```bash
   gunicorn -w 8 backend.app:app
   ```

4. **Enable async execution** (already enabled)

### For Low Latency

1. **Pre-build FAISS index**
2. **Warm up cache** with common queries
3. **Use local Redis** (not remote)
4. **Optimize database queries**

## Troubleshooting

### Backend won't start

```bash
# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip check

# Check logs
tail -f logs/agent_activity.log
```

### High memory usage

```bash
# Reduce connection pool
DB_POOL_SIZE=5

# Disable caching
REDIS_URL=memory://

# Use smaller model
# (Edit agents to use smaller LLM)
```

### Slow responses

```bash
# Enable caching
REDIS_URL=redis://localhost:6379/0

# Check cache hit rate
curl http://localhost:5000/api/cache-metrics

# Monitor agent execution times
# Check /metrics endpoint
```

### Rate limit errors

```bash
# Increase limits in app.py
# Or use admin token to bypass
Authorization: Bearer your_admin_token
```

## Backup & Recovery

### Backup Database

```bash
# SQLite backup
cp data/rfp_system.db data/rfp_system.db.backup

# Or use SQLite backup command
sqlite3 data/rfp_system.db ".backup data/backup.db"
```

### Backup FAISS Index

```bash
cp data/faiss_index.bin data/faiss_index.bin.backup
cp data/faiss_index_metadata.pkl data/faiss_index_metadata.pkl.backup
```

### Restore

```bash
# Restore database
cp data/rfp_system.db.backup data/rfp_system.db

# Restore FAISS index
cp data/faiss_index.bin.backup data/faiss_index.bin
cp data/faiss_index_metadata.pkl.backup data/faiss_index_metadata.pkl
```

## Scaling

### Horizontal Scaling

1. Use external Redis for shared cache
2. Use PostgreSQL instead of SQLite
3. Deploy multiple backend instances
4. Use load balancer (Nginx/HAProxy)
5. Share FAISS index via network storage

### Vertical Scaling

1. Increase RAM for larger FAISS index
2. Use faster CPU for embeddings
3. Use SSD for database
4. Increase worker processes

## Maintenance

### Regular Tasks

- **Daily**: Check logs for errors
- **Weekly**: Review metrics and performance
- **Monthly**: Update dependencies
- **Quarterly**: Security audit

### Updates

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Rebuild FAISS index
python scripts/rebuild_index.py

# Restart service
sudo systemctl restart rfp-agent
```

## Support

For issues and questions:
- Check logs: `logs/agent_activity.log`
- Review metrics: `http://localhost:5000/metrics`
- Check health: `http://localhost:5000/api/health`

## License

[Your License Here]
