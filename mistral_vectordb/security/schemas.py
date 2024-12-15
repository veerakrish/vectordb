"""Schema validation templates for VectorDB."""

VECTOR_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {
            "type": "object",
            "additionalProperties": True
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "permissions": {
            "type": "object",
            "properties": {
                "read": {"type": "array", "items": {"type": "string"}},
                "write": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    "required": ["text", "source", "timestamp"]
}

API_REQUEST_SCHEMA = {
    "add_vector": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "vector": {"type": "array", "items": {"type": "number"}},
            "metadata": {"$ref": "#/definitions/metadata"}
        },
        "required": ["id", "vector", "metadata"]
    },
    "search": {
        "type": "object",
        "properties": {
            "query_vector": {"type": "array", "items": {"type": "number"}},
            "query_text": {"type": "string"},
            "k": {"type": "integer", "minimum": 1},
            "filter": {"type": "object"},
            "hybrid_alpha": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["query_vector"]
    }
}

SECURITY_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "encryption": {
            "type": "object",
            "properties": {
                "algorithm": {"type": "string", "enum": ["AES-256-GCM", "ChaCha20-Poly1305"]},
                "key_rotation_days": {"type": "integer", "minimum": 1},
                "backup_encryption": {"type": "boolean"}
            }
        },
        "authentication": {
            "type": "object",
            "properties": {
                "token_expiry_minutes": {"type": "integer", "minimum": 1},
                "max_failed_attempts": {"type": "integer", "minimum": 1},
                "require_2fa": {"type": "boolean"}
            }
        },
        "audit": {
            "type": "object",
            "properties": {
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                "log_retention_days": {"type": "integer", "minimum": 1},
                "enable_audit_trail": {"type": "boolean"}
            }
        }
    }
}
