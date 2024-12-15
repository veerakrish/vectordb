"""Secure API server for VectorDB with enhanced security."""

from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime, timedelta
import jwt
import logging
import time
import uuid
from pathlib import Path

from ..security.crypto import SecureComm, SecureDataProcessor
from ..security.schemas import API_REQUEST_SCHEMA, VECTOR_METADATA_SCHEMA
from ..security.rate_limiter import RateLimiter
from ..security.audit import AuditLogger, EventCategory, EventSeverity
from ..store import VectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Secure VectorDB API")
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize security components
secure_comm = None
data_processor = None
vector_store = None
rate_limiter = None
audit_logger = None

class SearchRequest(BaseModel):
    query_vector: List[float]
    query_text: Optional[str] = None
    k: int = Field(default=10, gt=0)
    filter_criteria: Optional[Dict[str, Any]] = None
    hybrid_alpha: float = Field(default=0.5, ge=0, le=1)

class AddVectorRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any]

class TokenRequest(BaseModel):
    username: str
    password: str
    client_id: Optional[str] = None
    grant_type: str = "password"

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

def init_api(storage_dir: str, dim: int, api_key: str):
    """Initialize API with enhanced security components."""
    global secure_comm, data_processor, vector_store, rate_limiter, audit_logger
    
    secure_comm = SecureComm(api_key)
    data_processor = SecureDataProcessor(api_key)
    vector_store = VectorStore(
        storage_dir=storage_dir,
        dim=dim,
        api_key=api_key
    )
    
    # Initialize rate limiter with custom configuration
    rate_limiter = RateLimiter(
        default_rate_limit=100,
        block_threshold=5,
        block_duration=3600,
        whitelist_ips=["127.0.0.1"]  # Add trusted IPs
    )
    
    # Initialize audit logger
    audit_logger = AuditLogger(
        rotation_size_mb=100,
        retention_days=90,
        enable_encryption=True
    )

def get_client_ip(request: Request) -> str:
    """Get client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host

async def rate_limit_check(request: Request):
    """Check rate limit for request."""
    client_ip = get_client_ip(request)
    if not rate_limiter.check_rate_limit(client_ip, request.url.path):
        raise HTTPException(status_code=429, detail="Too many requests")

@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Middleware for request auditing."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000
        
        # Log successful request
        audit_logger.log_event(
            category=EventCategory.DATA_ACCESS,
            severity=EventSeverity.INFO,
            event_type="API_REQUEST",
            user_id=getattr(request.state, "user_id", "anonymous"),
            ip_address=get_client_ip(request),
            endpoint=request.url.path,
            request_id=request_id,
            details={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "status_code": response.status_code
            },
            status="success",
            duration_ms=duration
        )
        
        return response
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        
        # Log failed request
        audit_logger.log_event(
            category=EventCategory.DATA_ACCESS,
            severity=EventSeverity.ERROR,
            event_type="API_REQUEST",
            user_id=getattr(request.state, "user_id", "anonymous"),
            ip_address=get_client_ip(request),
            endpoint=request.url.path,
            request_id=request_id,
            details={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "error": str(e)
            },
            status="error",
            duration_ms=duration
        )
        
        raise

@app.post("/token")
async def login(
    request: Request,
    form_data: TokenRequest
):
    """OAuth2 compatible token login."""
    try:
        # Verify credentials (implement your auth logic here)
        user = authenticate_user(form_data.username, form_data.password)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        
        # Generate token
        access_token = secure_comm.generate_token(
            user_id=user["id"],
            permissions=user["permissions"],
            client_id=form_data.client_id
        )
        
        # Log successful login
        audit_logger.log_event(
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.INFO,
            event_type="USER_LOGIN",
            user_id=user["id"],
            ip_address=get_client_ip(request),
            endpoint="/token",
            request_id=request.state.request_id,
            details={
                "client_id": form_data.client_id,
                "grant_type": form_data.grant_type
            },
            status="success",
            duration_ms=0
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600
        )
        
    except Exception as e:
        # Log failed login
        audit_logger.log_event(
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.WARNING,
            event_type="USER_LOGIN",
            user_id=form_data.username,
            ip_address=get_client_ip(request),
            endpoint="/token",
            request_id=request.state.request_id,
            details={
                "error": str(e),
                "client_id": form_data.client_id,
                "grant_type": form_data.grant_type
            },
            status="error",
            duration_ms=0
        )
        raise

@app.post("/vectors/add", dependencies=[Depends(rate_limit_check)])
async def add_vector(
    request: Request,
    vector_request: AddVectorRequest,
    token_payload: Dict = Depends(verify_token)
):
    """Add vector with enhanced security."""
    try:
        # Verify request integrity
        secure_comm.verify_request(vector_request.dict())
        
        # Validate schema
        data_processor.validate_schema(vector_request.dict(), "add_vector")
        
        # Check permissions
        if not has_write_permission(token_payload, vector_request.metadata):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Add vector
        vector_store.add_item(
            vector_request.id,
            np.array(vector_request.vector),
            vector_request.metadata
        )
        
        return secure_comm.secure_response({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error adding vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectors/search", dependencies=[Depends(rate_limit_check)])
async def search_vectors(
    request: Request,
    search_request: SearchRequest,
    token_payload: Dict = Depends(verify_token)
):
    """Search vectors with enhanced security."""
    try:
        # Verify request integrity
        secure_comm.verify_request(search_request.dict())
        
        # Validate schema
        data_processor.validate_schema(search_request.dict(), "search")
        
        # Perform search with permission filtering
        results = vector_store.search(
            query_vector=np.array(search_request.query_vector),
            query_text=search_request.query_text,
            k=search_request.k,
            filter_func=lambda x: has_read_permission(token_payload, x),
            hybrid_alpha=search_request.hybrid_alpha
        )
        
        return secure_comm.secure_response({
            "results": [
                {
                    "id": r[0],
                    "score": float(r[1]),
                    "metadata": r[2]
                }
                for r in results
            ]
        })
        
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit/logs")
async def get_audit_logs(
    request: Request,
    token_payload: Dict = Depends(verify_token),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None
):
    """Query audit logs with filtering."""
    # Check admin permission
    if not is_admin(token_payload):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        events = audit_logger.query_logs(
            start_time=start_time,
            end_time=end_time,
            category=EventCategory[category] if category else None,
            severity=EventSeverity[severity] if severity else None,
            user_id=user_id,
            status=status
        )
        
        return secure_comm.secure_response({
            "events": [asdict(event) for event in events]
        })
        
    except Exception as e:
        logger.error(f"Error querying audit logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def has_read_permission(token_payload: Dict, metadata: Dict) -> bool:
    """Check if user has read permission for the vector."""
    user_id = token_payload.get("sub")
    permissions = metadata.get("permissions", {})
    return (
        user_id in permissions.get("read", []) or
        user_id in permissions.get("write", [])
    )

def has_write_permission(token_payload: Dict, metadata: Dict) -> bool:
    """Check if user has write permission for the vector."""
    user_id = token_payload.get("sub")
    permissions = metadata.get("permissions", {})
    return user_id in permissions.get("write", [])

def is_admin(token_payload: Dict) -> bool:
    """Check if user is admin."""
    return token_payload.get("role") == "admin"

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token and permissions."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, secure_comm.get_public_key(), algorithms=["RS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
