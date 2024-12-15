# Mistral VectorDB

A high-performance vector database with secure storage, advanced document processing, and RAG features, powered by Mistral AI.

## Features

- **AI-Powered Security** using Mistral AI
- **High-Performance Vector Search**
- **Advanced Document Processing**
- **Real-time Updates**
- **API Server with Rate Limiting**
- **Monitoring and Metrics**
- **Hybrid Search**
- **MMR Diversity**
- **Cross-encoder Re-ranking**

## Installation

```bash
# Install with pip
pip install mistral-vectordb

# Or install from source
git clone https://github.com/veerakrish/mistral-vectordb.git
cd mistral-vectordb
pip install -e .
```

## Quick Start

```python
from mistral_vectordb import Collection
from mistral_vectordb.embeddings import MistralEmbedding

# Initialize collection
collection = Collection(
    name="my_collection",
    embedding_function=MistralEmbedding().embed
)

# Add documents
collection.add(
    documents=["Document 1", "Document 2"],
    metadatas=[{"source": "web"}, {"source": "file"}]
)

# Query
results = collection.query(
    query_texts=["Search query"],
    n_results=2
)
```

## Secure API Server

### Starting the Server

```python
from mistral_vectordb.server import start_server

# Start server
start_server(host="0.0.0.0", port=8000)
```

### Client Usage

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Add documents securely
documents = [
    {
        "text": "Sensitive document 1",
        "metadata": {"confidential": True}
    }
]
response = requests.post(
    f"{API_URL}/api/collections/secure_docs/add",
    headers=HEADERS,
    json=documents
)

# Query with security
query = {
    "query_text": "Find confidential",
    "n_results": 5,
    "where": {"confidential": True}
}
response = requests.post(
    f"{API_URL}/api/collections/secure_docs/query",
    headers=HEADERS,
    json=query
)
```

## Security Features

### 1. AI-Powered Security

```python
from mistral_vectordb.security import SecureStorage, SecureComm, SecureDataProcessor

# Initialize security components
secure_storage = SecureStorage(mistral_api_key)
secure_comm = SecureComm(mistral_api_key)
secure_processor = SecureDataProcessor(mistral_api_key)

# Secure data storage
encrypted_data = secure_storage.encrypt_data(sensitive_data)
decrypted_data = secure_storage.decrypt_data(encrypted_data)

# Verify requests
is_safe = secure_comm.verify_request(request_data)

# Sanitize and validate data
clean_data = secure_processor.sanitize_input(user_data)
is_valid = secure_processor.validate_schema(clean_data, "document")
```

### 2. Batch Processing

```python
# Submit batch job
response = requests.post(
    f"{API_URL}/api/batch/submit",
    headers=HEADERS,
    json={
        "collection_name": "large_docs",
        "documents": large_document_list
    }
)
job_id = response.json()["job_id"]

# Check status
status = requests.get(
    f"{API_URL}/api/batch/{job_id}/status",
    headers=HEADERS
)
print(status.json())  # {"status": "processing", "progress": 0.75}
```

### 3. Monitoring

```python
# Health check
response = requests.get(f"{API_URL}/health")

# System metrics (admin only)
response = requests.get(
    f"{API_URL}/system",
    auth=("admin", "password")
)

# Prometheus metrics
response = requests.get(
    f"{API_URL}/metrics",
    auth=("admin", "password")
)
```

## Rate Limits

```plaintext
Endpoint                    | Limit
---------------------------|----------------
/api/collections/add       | 100/minute
/api/collections/query     | 200/minute
/api/collections/get       | 50/minute
/api/collections/delete    | 20/minute
/api/generate_key          | 5/minute
/api/batch/submit          | 10/minute
/health                    | 60/minute
/system                    | 10/minute
```

## Authentication Methods

### 1. JWT Token

```python
# Generate API key
response = requests.post(
    f"{API_URL}/api/generate_key",
    json={"name": "client1", "expiry_days": 30}
)
api_key = response.json()["api_key"]
```

### 2. OAuth with GitHub

```python
# Redirect users to
auth_url = f"{API_URL}/login/github"

# Get API token from callback
# The server automatically converts GitHub OAuth token to API token
```

### 3. Basic Auth (Admin Only)

```python
# Access admin endpoints
response = requests.get(
    f"{API_URL}/system",
    auth=("admin", "admin_password")
)
```

## Security Best Practices

1. **API Keys**:
   - Store securely in environment variables
   - Rotate regularly
   - Set appropriate expiration

2. **Sensitive Data**:
   - Always use HTTPS in production
   - Encrypt metadata
   - Use secure storage

3. **Monitoring**:
   - Watch rate limit violations
   - Monitor error rates
   - Track system metrics

## Production Deployment

1. **Environment Setup**:
```bash
# Set required environment variables
export MISTRAL_API_KEY="your-key"
export GITHUB_CLIENT_ID="your-id"
export GITHUB_CLIENT_SECRET="your-secret"
```

2. **Prometheus Monitoring**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vectordb'
    static_configs:
      - targets: ['localhost:8000']
```

3. **Secure Server Start**:
```python
start_server(
    host="0.0.0.0",
    port=443,
    ssl_keyfile="path/to/key.pem",
    ssl_certfile="path/to/cert.pem"
)
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Example Use Cases

### 1. Secure Document Management System

```python
from mistral_vectordb import Collection
from mistral_vectordb.security import SecureStorage, HIPAA_Compliance

class MedicalRecordSystem:
    def __init__(self):
        self.secure = SecureStorage(mistral_api_key)
        self.hipaa = HIPAA_Compliance()
        self.collection = Collection(
            name="medical_records",
            embedding_function=MistralEmbedding().embed
        )
    
    def add_patient_record(self, record):
        # HIPAA compliance check
        if not self.hipaa.validate_record(record):
            raise ValueError("Record does not meet HIPAA requirements")
        
        # PHI encryption
        encrypted_phi = self.secure.encrypt_data({
            "patient_id": record["patient_id"],
            "ssn": record["ssn"],
            "dob": record["dob"]
        })
        
        # Store with access controls
        self.collection.add(
            documents=[record["medical_history"]],
            metadatas=[{
                "phi": encrypted_phi,
                "department": record["department"],
                "access_level": "medical_staff"
            }]
        )
    
    def search_records(self, query, staff_credentials):
        if not self.hipaa.verify_access(staff_credentials):
            raise PermissionError("Unauthorized access")
        
        return self.collection.query(
            query_texts=[query],
            where={"access_level": staff_credentials["level"]}
        )
```

### 2. Real-time Document Processing Pipeline

```python
from mistral_vectordb.server import start_server
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        # Submit document for processing
        asyncio.create_task(self.process_document(event.src_path))
    
    async def process_document(self, filepath):
        # Submit batch job
        response = requests.post(
            f"{self.api_url}/api/batch/submit",
            headers=self.headers,
            json={
                "collection_name": "realtime_docs",
                "documents": [{"filepath": filepath}]
            }
        )
        
        # Monitor progress
        job_id = response.json()["job_id"]
        while True:
            status = requests.get(
                f"{self.api_url}/api/batch/{job_id}/status",
                headers=self.headers
            ).json()
            
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(1)

# Start monitoring
observer = Observer()
observer.schedule(
    DocumentHandler(API_URL, API_KEY),
    path="watched_directory",
    recursive=False
)
observer.start()
```

### 3. Multi-Modal Search System

```python
from mistral_vectordb import Collection
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel

class MultiModalSearch:
    def __init__(self):
        self.collection = Collection("multimodal")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def process_image(self, image_path):
        # Extract text with OCR
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        # Get image embedding
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.clip.get_image_features(**inputs)
        
        return {
            "text": text,
            "image_embedding": image_features.tolist()[0],
            "metadata": {
                "type": "image",
                "path": image_path
            }
        }
    
    def add_image(self, image_path):
        processed = self.process_image(image_path)
        self.collection.add(
            documents=[processed["text"]],
            metadatas=[processed["metadata"]],
            embeddings=[processed["image_embedding"]]
        )
    
    def search(self, query, mode="text"):
        if mode == "text":
            return self.collection.query(
                query_texts=[query]
            )
        elif mode == "image":
            # Process query image
            processed = self.process_image(query)
            return self.collection.query(
                query_embeddings=[processed["image_embedding"]]
            )
```

## API Endpoints Reference

### Vector Operations

```python
import requests
import os

base_url = "http://localhost:8000"
headers = {
    "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
    "Content-Type": "application/json"
}

# Batch add vectors with metadata
response = requests.post(
    f"{base_url}/vectors/batch_add",
    headers=headers,
    json={
        "documents": [
            "First document",
            "Second document",
            "Third document"
        ],
        "metadata": [
            {"category": "A", "tags": ["tag1"]},
            {"category": "B", "tags": ["tag2"]},
            {"category": "C", "tags": ["tag3"]}
        ]
    }
)

# Delete vectors by IDs
response = requests.delete(
    f"{base_url}/vectors/delete",
    headers=headers,
    json={
        "ids": ["vec_1", "vec_2"]
    }
)

# Update vector metadata
response = requests.put(
    f"{base_url}/vectors/update",
    headers=headers,
    json={
        "id": "vec_1",
        "metadata": {"category": "updated", "tags": ["new_tag"]}
    }
)

# Advanced search with filters
response = requests.post(
    f"{base_url}/vectors/search",
    headers=headers,
    json={
        "query_text": "example query",
        "n_results": 5,
        "filters": {
            "category": "A",
            "tags": {"$in": ["tag1", "tag2"]},
            "date": {"$gt": "2024-01-01"}
        },
        "include_metadata": True,
        "include_distances": True
    }
)

# Hybrid search (combine semantic and keyword search)
response = requests.post(
    f"{base_url}/vectors/hybrid_search",
    headers=headers,
    json={
        "query_text": "example query",
        "keyword_weight": 0.3,
        "semantic_weight": 0.7,
        "n_results": 5
    }
)
```

### Collection Management

```python
# Create new collection
response = requests.post(
    f"{base_url}/collections/create",
    headers=headers,
    json={
        "name": "my_collection",
        "dimension": 768,
        "metric": "cosine"
    }
)

# List collections
response = requests.get(
    f"{base_url}/collections/list",
    headers=headers
)

# Get collection stats
response = requests.get(
    f"{base_url}/collections/my_collection/stats",
    headers=headers
)

# Delete collection
response = requests.delete(
    f"{base_url}/collections/my_collection",
    headers=headers
)
```

### Administration

```python
# Get server stats
response = requests.get(
    f"{base_url}/admin/stats",
    headers=headers
)

# Get audit logs
response = requests.get(
    f"{base_url}/admin/audit_logs",
    headers=headers,
    params={
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-12-31T23:59:59Z",
        "severity": "ERROR",
        "limit": 100
    }
)

# Update rate limits
response = requests.put(
    f"{base_url}/admin/rate_limits",
    headers=headers,
    json={
        "default_rate": 200,
        "ip_whitelist": ["192.168.1.100"]
    }
)
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Connection Refused**
```
ConnectionRefusedError: [Errno 61] Connection refused
```
- Check if the server is running
- Verify the correct host and port
- Ensure no firewall is blocking the connection
- Try using `localhost` instead of `0.0.0.0`

2. **Authentication Errors**
```
{"error": "Invalid API key"}
```
- Verify API key is set correctly in environment
- Check if API key has expired
- Ensure Authorization header is formatted correctly
- Try regenerating the API key

3. **Rate Limiting Issues**
```
{"error": "Rate limit exceeded"}
```
- Check current rate limit configuration
- Consider using batch operations
- Implement exponential backoff
- Request rate limit increase if needed

4. **Memory Issues**
```
MemoryError: Unable to allocate array
```
- Reduce batch size for vector operations
- Monitor server memory usage
- Consider upgrading server resources
- Enable memory optimization settings

5. **Index Corruption**
```
{"error": "Index corrupted"}
```
- Backup data immediately
- Check disk space
- Rebuild index from backup
- Enable periodic index verification

### Debugging Tips

1. **Enable Debug Logging**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('mistral_vectordb')
```

2. **Check Server Logs**
```bash
# View real-time logs
tail -f ./vectordb_data/logs/server.log

# Search for errors
grep "ERROR" ./vectordb_data/logs/server.log
```

3. **Verify Configuration**
```python
# Print current configuration
response = requests.get(
    f"{base_url}/admin/config",
    headers=headers
)
print(response.json())
```

## Production Deployment Recommendations

### 1. Infrastructure Setup

```yaml
# production_config.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 8
  ssl_enabled: true
  ssl_cert: /path/to/cert.pem
  ssl_key: /path/to/key.pem
  max_request_size: 10MB

storage:
  directory: /data/vectordb
  backup_enabled: true
  backup_interval: 24h
  backup_retention: 7d

security:
  rate_limit:
    default_rate: 1000
    burst_rate: 2000
  audit:
    enabled: true
    log_dir: /var/log/vectordb
    rotation_size: 100MB
    retention_days: 90
  encryption:
    enabled: true
    key_rotation: 30d
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m vectordb
USER vectordb

CMD ["vectordb-server", "--config", "production_config.yaml"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  vectordb:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - vectordb_data:/data/vectordb
      - vectordb_logs:/var/log/vectordb
    environment:
      - MISTRAL_API_KEY=${MISTRAL_API_KEY}
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### 3. Performance Optimization

1. **Memory Management**
- Use memory-mapped files for large indices
- Implement periodic garbage collection
- Monitor memory usage with Prometheus/Grafana

2. **CPU Optimization**
- Enable multi-threading for search operations
- Use batch processing for large updates
- Configure optimal number of workers

3. **Disk I/O**
- Use SSD storage for better performance
- Implement proper backup strategy
- Monitor disk usage and IOPS

### 4. Security Best Practices

1. **Network Security**
- Use reverse proxy (nginx/traefik)
- Enable SSL/TLS
- Implement IP whitelisting
- Use secure headers

2. **Access Control**
- Implement role-based access
- Regular key rotation
- Audit logging
- Rate limiting

3. **Data Protection**
- Regular backups
- Data encryption at rest
- Secure key management
- Compliance monitoring

### 5. Monitoring Setup

```python
# Prometheus metrics endpoint
response = requests.get(
    f"{base_url}/metrics",
    headers=headers
)

# Health check endpoint with detailed status
response = requests.get(
    f"{base_url}/health/detailed",
    headers=headers
)
```

### 6. Backup Strategy

```bash
# Automated backup script
#!/bin/bash
backup_dir="/backup/vectordb"
date=$(date +%Y%m%d)

# Stop writes
curl -X POST http://localhost:8000/admin/maintenance/start

# Create backup
tar -czf "$backup_dir/vectordb_$date.tar.gz" /data/vectordb/

# Resume writes
curl -X POST http://localhost:8000/admin/maintenance/end

# Cleanup old backups
find "$backup_dir" -type f -mtime +7 -delete
