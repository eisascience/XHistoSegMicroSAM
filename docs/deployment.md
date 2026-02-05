# Deployment Guide for XHalo Path Analyzer

This guide provides instructions for deploying the Halo AI Workflow application in various environments.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud](#streamlit-cloud)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [Production Considerations](#production-considerations)

## Local Deployment

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   
   Or use the CLI:
   ```bash
   xhalo-analyzer web
   ```

3. **Access the application:**
   - Open your browser to http://localhost:8501

### Custom Configuration

Create or modify `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "localhost"
headless = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

## Streamlit Cloud

Streamlit Cloud provides free hosting for Streamlit applications.

### Steps:

1. **Push your code to GitHub:**
   ```bash
   git push origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Deploy your repository

3. **Configure secrets (if needed):**
   - Add API keys in the Streamlit Cloud dashboard
   - Access via `st.secrets` in your code

### Streamlit Cloud Configuration

Create `.streamlit/secrets.toml` (not committed):
```toml
[halo]
api_url = "https://your-halo-instance/graphql"
api_key = "your-api-key-here"

[medsam]
model_path = "/path/to/model.pth"
```

## Docker Deployment

### Dockerfile

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  xhalo-analyzer:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - HALO_API_URL=${HALO_API_URL}
      - HALO_API_KEY=${HALO_API_KEY}
    restart: unless-stopped
```

### Build and Run:

```bash
# Build image
docker build -t xhalo-analyzer .

# Run container
docker run -p 8501:8501 xhalo-analyzer

# Or use docker-compose
docker-compose up -d
```

## Cloud Platforms

### AWS EC2

1. **Launch EC2 instance:**
   - Choose Ubuntu 22.04 LTS
   - Instance type: t3.medium or larger (for GPU: p3.2xlarge, for Apple Silicon alternative: consider AWS Graviton instances)
   - Open port 8501 in security group

2. **Setup:**
   ```bash
   ssh ubuntu@your-instance-ip
   
   # Install dependencies
   sudo apt update
   sudo apt install -y python3-pip git
   
   # Clone and install
   git clone https://github.com/eisascience/XHaloPathAnalyzer.git
   cd XHaloPathAnalyzer
   pip3 install -r requirements.txt
   pip3 install -e .
   
   # Run with nohup
   nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
   ```

3. **Use systemd for production:**
   Create `/etc/systemd/system/xhalo.service`:
   ```ini
   [Unit]
   Description=XHalo Path Analyzer
   After=network.target
   
   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/XHaloPathAnalyzer
   ExecStart=/usr/local/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable and start:
   ```bash
   sudo systemctl enable xhalo
   sudo systemctl start xhalo
   ```

### AWS ECS/Fargate

1. **Push Docker image to ECR:**
   ```bash
   aws ecr create-repository --repository-name xhalo-analyzer
   docker tag xhalo-analyzer:latest <account-id>.dkr.ecr.<region>.amazonaws.com/xhalo-analyzer:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/xhalo-analyzer:latest
   ```

2. **Create ECS task definition and service**
3. **Configure Application Load Balancer**

### Google Cloud Run

1. **Build and push to Container Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/your-project/xhalo-analyzer
   ```

2. **Deploy:**
   ```bash
   gcloud run deploy xhalo-analyzer \
     --image gcr.io/your-project/xhalo-analyzer \
     --platform managed \
     --port 8501 \
     --allow-unauthenticated
   ```

### Azure Web Apps

1. **Create App Service:**
   ```bash
   az webapp create \
     --resource-group myResourceGroup \
     --plan myAppServicePlan \
     --name xhalo-analyzer \
     --deployment-container-image-name xhalo-analyzer:latest
   ```

2. **Configure port and settings**

### Heroku

1. **Create Heroku app:**
   ```bash
   heroku create xhalo-analyzer
   ```

2. **Add Procfile:**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

## Production Considerations

### Security

1. **Environment Variables:**
   - Store sensitive data in environment variables
   - Use secrets management (AWS Secrets Manager, etc.)

2. **HTTPS:**
   - Use reverse proxy (nginx, Apache) with SSL
   - Or use cloud platform SSL termination

3. **Authentication:**
   - Implement authentication if needed
   - Use OAuth or SSO integration

### Performance

1. **Caching:**
   ```python
   @st.cache_data
   def load_large_data():
       # Expensive operation
       pass
   ```

2. **Resource Limits:**
   - Set memory and CPU limits in Docker
   - Monitor resource usage

3. **GPU Acceleration:**
   - For NVIDIA GPUs: Use GPU instances (p3.2xlarge) for MedSAM inference and install CUDA and PyTorch with GPU support
   - For Apple Silicon: PyTorch 2.6.0+ automatically supports MPS acceleration

### Monitoring

1. **Logging:**
   - Configure application logging
   - Use centralized logging (CloudWatch, Stackdriver)

2. **Health Checks:**
   - Implement health check endpoint
   - Monitor application availability

3. **Metrics:**
   - Track processing times
   - Monitor API usage

### Scalability

1. **Horizontal Scaling:**
   - Deploy multiple instances behind load balancer
   - Use container orchestration (Kubernetes)

2. **Database:**
   - Store annotations in database
   - Cache frequently accessed data

3. **Message Queue:**
   - Use queue for batch processing
   - Implement asynchronous processing

### Backup and Recovery

1. **Data Backup:**
   - Regular backups of models and configurations
   - Version control for code

2. **Disaster Recovery:**
   - Multi-region deployment for critical applications
   - Documented recovery procedures

## Example Production Stack

```
Internet
   ↓
[Load Balancer with SSL]
   ↓
[Multiple XHalo Instances]
   ↓
[Halo API] ← → [Redis Cache]
   ↓
[PostgreSQL Database]
```

## Support

For deployment issues:
- Check the [GitHub Issues](https://github.com/eisascience/XHaloPathAnalyzer/issues)
- Review [Streamlit Documentation](https://docs.streamlit.io/)
- Consult cloud platform documentation

## License

MIT License - See LICENSE file for details
