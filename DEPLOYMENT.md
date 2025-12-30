# Evidence Suite - Deployment Guide

This guide covers deploying Evidence Suite in production environments.

## Prerequisites

- Docker and Docker Compose
- PostgreSQL 16+
- Redis 7+
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for accelerated analysis)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/AuraquanTech/evidence-suite.git
cd evidence-suite

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings (see Configuration section)

# Start services
docker-compose up -d

# Check health
curl http://localhost:8000/ready
```

## Configuration

### Required Environment Variables

These MUST be set in production:

```bash
# Environment mode - CRITICAL: Set to 'production' for production deployments
EVIDENCE_SUITE_ENV=production

# Database (PostgreSQL only in production)
DATABASE_URL=postgresql://user:password@host:5432/evidence_suite

# JWT Secret - CRITICAL: Generate a unique secret
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
API_JWT_SECRET=your-unique-secret-here

# Encryption key for evidence at rest - CRITICAL for compliance
# Generate with: python -c "import secrets; import base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
EVIDENCE_ENCRYPTION_KEY=your-base64-encoded-32-byte-key
```

### Optional Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_JWT_EXPIRE_MINUTES=30
API_CORS_ORIGINS=["https://your-domain.com"]

# Redis (for caching and rate limiting)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Database Connection Pool
DB_MAX_RETRIES=5
DB_RETRY_DELAY=1.0
DB_RETRY_MAX_DELAY=30.0
DB_STARTUP_TIMEOUT=60

# File Storage
EVIDENCE_STORAGE_PATH=/data/evidence
MAX_UPLOAD_SIZE_MB=500

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# Logging
LOG_LEVEL=INFO

# Shutdown
SHUTDOWN_GRACE_PERIOD=10
```

## Docker Deployment

### Production Docker Compose

```yaml
version: '3.8'

services:
  api:
    image: evidence-suite:latest
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - EVIDENCE_SUITE_ENV=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/evidence_suite
      - REDIS_HOST=redis
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=evidence_suite
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  worker:
    image: evidence-suite:latest
    command: arq worker.tasks.WorkerSettings
    environment:
      - EVIDENCE_SUITE_ENV=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/evidence_suite
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### GPU Support

For GPU-accelerated analysis, use the GPU Dockerfile:

```yaml
services:
  api:
    image: evidence-suite:gpu
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - ENABLE_GPU=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evidence-suite-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: evidence-suite-api
  template:
    metadata:
      labels:
        app: evidence-suite-api
    spec:
      containers:
      - name: api
        image: evidence-suite:latest
        ports:
        - containerPort: 8000
        env:
        - name: EVIDENCE_SUITE_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: evidence-suite-secrets
              key: database-url
        - name: API_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: evidence-suite-secrets
              key: jwt-secret
        - name: EVIDENCE_ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: evidence-suite-secrets
              key: encryption-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: evidence-suite-api
spec:
  selector:
    app: evidence-suite-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Database Migrations

Run migrations before starting the application:

```bash
# Using Alembic
alembic upgrade head

# Or via Docker
docker-compose exec api alembic upgrade head
```

## Monitoring

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'evidence-suite'
    static_configs:
      - targets: ['evidence-suite-api:8000']
    metrics_path: '/metrics/prometheus'
    scrape_interval: 15s
```

### Health Check Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/live` | Liveness probe | Always 200 if running |
| `/ready` | Readiness probe | 200 when ready, 503 when not |
| `/health` | Detailed health check | JSON with component status |
| `/health/db` | Database health | Detailed DB metrics |
| `/metrics` | JSON metrics | Application metrics |
| `/metrics/prometheus` | Prometheus format | Prometheus-compatible metrics |

## Security Checklist

Before going to production:

- [ ] Set `EVIDENCE_SUITE_ENV=production`
- [ ] Generate unique `API_JWT_SECRET` (min 32 characters)
- [ ] Generate unique `EVIDENCE_ENCRYPTION_KEY`
- [ ] Configure `DATABASE_URL` (PostgreSQL only)
- [ ] Set `REDIS_PASSWORD`
- [ ] Configure `API_CORS_ORIGINS` for your domain
- [ ] Enable HTTPS via reverse proxy (nginx, traefik)
- [ ] Configure firewall rules
- [ ] Set up log aggregation
- [ ] Configure backup strategy for database and evidence storage

## Backup Strategy

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U postgres evidence_suite > backup.sql

# Restore
psql -h localhost -U postgres evidence_suite < backup.sql
```

### Evidence Storage Backup

```bash
# Backup evidence files
tar -czvf evidence_backup.tar.gz /data/evidence

# Restore
tar -xzvf evidence_backup.tar.gz -C /
```

## Troubleshooting

### Application Won't Start

1. Check logs: `docker-compose logs api`
2. Verify environment variables are set
3. Check database connectivity: `curl http://localhost:8000/health/db`
4. Check Redis connectivity

### Database Connection Issues

```bash
# Test connection
docker-compose exec api python -c "from core.database.session import test_connection; import asyncio; print(asyncio.run(test_connection()))"
```

### High Memory Usage

- Reduce worker pool size
- Decrease batch size for analysis
- Enable swap if needed

### Slow Analysis

- Check GPU availability
- Increase worker replicas
- Check Redis connectivity for caching

## Support

For issues and feature requests, please visit:
https://github.com/AuraquanTech/evidence-suite/issues
