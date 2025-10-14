from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
"""
Data Privacy Service
Enterprise-grade data privacy, anonymization, and compliance management
"""

import asyncio
import logging
import signal
import hashlib
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncpg
from dataclasses import dataclass
from enum import Enum

# Import tracing setup
import sys
import os
sys.path.append('/app')
from tracing_setup import setup_tracing, get_tracer, trace_request, add_span_attributes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
class PrivacyLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DataType(Enum):
    PII = "pii"  # Personally Identifiable Information
    SENSITIVE = "sensitive"
    FINANCIAL = "financial"
    HEALTH = "health"
    BUSINESS = "business"

class AnonymizationMethod(Enum):
    HASH = "hash"
    MASK = "mask"
    REDACT = "redact"
    PSEUDONYMIZE = "pseudonymize"
    GENERALIZE = "generalize"

# Pydantic models
class PrivacyPolicy(BaseModel):
    policy_id: str
    name: str
    description: str
    privacy_level: PrivacyLevel
    data_types: List[DataType]
    retention_days: int
    anonymization_required: bool
    anonymization_method: Optional[AnonymizationMethod] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class DataClassification(BaseModel):
    data_id: str
    data_type: DataType
    privacy_level: PrivacyLevel
    contains_pii: bool
    pii_fields: List[str] = []
    sensitivity_score: float = Field(ge=0.0, le=1.0)
    classification_reason: str
    classified_at: datetime = Field(default_factory=datetime.now)

class AnonymizationRequest(BaseModel):
    data: Dict[str, Any]
    policy_id: str
    fields_to_anonymize: List[str] = []
    anonymization_method: AnonymizationMethod = AnonymizationMethod.HASH

class AnonymizationResponse(BaseModel):
    anonymized_data: Dict[str, Any]
    anonymized_fields: List[str]
    privacy_score: float
    compliance_status: str
    anonymization_log: List[Dict[str, Any]]

class ComplianceReport(BaseModel):
    report_id: str
    total_datasets: int
    compliant_datasets: int
    non_compliant_datasets: int
    compliance_percentage: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime = Field(default_factory=datetime.now)

# Data classes
@dataclass
class PIIPattern:
    name: str
    pattern: str
    data_type: DataType
    sensitivity: float

class DataPrivacyService:
    """Enterprise data privacy and compliance management"""
    
    def __init__(self):
        self.db_url = "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"
        self.conn_pool = None
        self.pii_patterns = self._load_pii_patterns()
        self.privacy_policies = {}
        self.start_time = datetime.now()
        
    def _load_pii_patterns(self) -> List[PIIPattern]:
        """Load PII detection patterns"""
        return [
            PIIPattern("email", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', DataType.PII, 0.9),
            PIIPattern("phone", r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', DataType.PII, 0.8),
            PIIPattern("ssn", r'\b\d{3}-\d{2}-\d{4}\b', DataType.PII, 1.0),
            PIIPattern("credit_card", r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', DataType.FINANCIAL, 1.0),
            PIIPattern("ip_address", r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', DataType.PII, 0.6),
            PIIPattern("name", r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', DataType.PII, 0.7),
            PIIPattern("address", r'\b\d+\s+[A-Za-z0-9\s,.-]+\b', DataType.PII, 0.8),
        ]
    
    async def initialize(self):
        """Initialize database and load policies"""
        try:
            # Initialize PostgreSQL connection pool
            self.conn_pool = await asyncpg.create_pool(self.db_url)
            await self._create_tables()
            await self._load_privacy_policies()
            
            logger.info("✅ Data Privacy Service initialized")
            
    except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="data_privacy_initialization",
                additional_context={"service": "data-privacy"}
            )
        raise
    
    async def close(self):
        """Close connections"""
        if self.conn_pool:
            await self.conn_pool.close()
        logger.info("✅ Data Privacy Service closed")
    
    async def _create_tables(self):
        """Create privacy-related tables"""
        async with self.conn_pool.acquire() as conn:
            # Privacy policies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS privacy_policies (
                    policy_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    privacy_level VARCHAR(50) NOT NULL,
                    data_types JSONB,
                    retention_days INTEGER NOT NULL,
                    anonymization_required BOOLEAN DEFAULT FALSE,
                    anonymization_method VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Data classifications table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS data_classifications (
                    id SERIAL PRIMARY KEY,
                    data_id VARCHAR(255) NOT NULL,
                    data_type VARCHAR(50) NOT NULL,
                    privacy_level VARCHAR(50) NOT NULL,
                    contains_pii BOOLEAN DEFAULT FALSE,
                    pii_fields JSONB,
                    sensitivity_score DOUBLE PRECISION,
                    classification_reason TEXT,
                    classified_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Anonymization logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS anonymization_logs (
                    id SERIAL PRIMARY KEY,
                    data_id VARCHAR(255) NOT NULL,
                    policy_id VARCHAR(255) NOT NULL,
                    anonymized_fields JSONB,
                    anonymization_method VARCHAR(50),
                    privacy_score DOUBLE PRECISION,
                    compliance_status VARCHAR(50),
                    anonymized_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            logger.info("✅ Data privacy tables created")
    
    async def _load_privacy_policies(self):
        """Load privacy policies from database"""
        try:
            async with self.conn_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM privacy_policies")
                
                for row in rows:
                    policy = PrivacyPolicy(
                        policy_id=row['policy_id'],
                        name=row['name'],
                        description=row['description'],
                        privacy_level=PrivacyLevel(row['privacy_level']),
                        data_types=[DataType(dt) for dt in json.loads(row['data_types'])],
                        retention_days=row['retention_days'],
                        anonymization_required=row['anonymization_required'],
                        anonymization_method=AnonymizationMethod(row['anonymization_method']) if row['anonymization_method'] else None,
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    self.privacy_policies[policy.policy_id] = policy
                
                logger.info(f"✅ Loaded {len(self.privacy_policies)} privacy policies")
                
        except Exception as e:
            logger.error(f"❌ Failed to load privacy policies: {e}")
    
    async def classify_data(self, data: Dict[str, Any], data_id: str) -> DataClassification:
        """Classify data for privacy compliance"""
        try:
            # Detect PII and sensitive information
            pii_fields = []
            sensitivity_score = 0.0
            max_sensitivity = 0.0
            
            # Convert data to string for pattern matching
            data_str = json.dumps(data, default=str).lower()
            
            for pattern in self.pii_patterns:
                matches = re.findall(pattern.pattern, data_str, re.IGNORECASE)
                if matches:
                    pii_fields.append(pattern.name)
                    max_sensitivity = max(max_sensitivity, pattern.sensitivity)
            
            # Calculate overall sensitivity score
            if pii_fields:
                sensitivity_score = max_sensitivity
                contains_pii = True
                data_type = DataType.PII
                privacy_level = PrivacyLevel.RESTRICTED
            else:
                contains_pii = False
                data_type = DataType.BUSINESS
                privacy_level = PrivacyLevel.INTERNAL
            
            # Create classification
            classification = DataClassification(
                data_id=data_id,
                data_type=data_type,
                privacy_level=privacy_level,
                contains_pii=contains_pii,
                pii_fields=pii_fields,
                sensitivity_score=sensitivity_score,
                classification_reason=f"Detected {len(pii_fields)} PII patterns: {', '.join(pii_fields)}" if pii_fields else "No PII detected"
            )
            
            # Store classification
            await self._store_classification(classification)
            
            return classification
            
        except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="classify_data",
                additional_context={"data_id": data_id, "data_keys": list(data.keys()) if isinstance(data, dict) else "unknown"}
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_classification(self, classification: DataClassification):
        """Store data classification in database"""
        try:
            async with self.conn_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO data_classifications 
                    (data_id, data_type, privacy_level, contains_pii, pii_fields, 
                     sensitivity_score, classification_reason, classified_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                    classification.data_id,
                    classification.data_type.value,
                    classification.privacy_level.value,
                    classification.contains_pii,
                    json.dumps(classification.pii_fields),
                    classification.sensitivity_score,
                    classification.classification_reason,
                    classification.classified_at
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to store classification: {e}")
    
    async def anonymize_data(self, request: AnonymizationRequest) -> AnonymizationResponse:
        """Anonymize data according to privacy policy"""
        try:
            # Get privacy policy or use default
            if request.policy_id not in self.privacy_policies:
                # Create a default policy if none exists
                default_policy = PrivacyPolicy(
                    policy_id="default",
                    name="Default Policy",
                    description="Default privacy policy",
                    privacy_level=PrivacyLevel.INTERNAL,
                    data_types=[DataType.PII],
                    retention_days=365,
                    anonymization_required=True,
                    anonymization_method=AnonymizationMethod.HASH
                )
                self.privacy_policies["default"] = default_policy
            
            policy = self.privacy_policies[request.policy_id]
            anonymized_data = request.data.copy()
            anonymized_fields = []
            anonymization_log = []
            
            # Determine fields to anonymize
            fields_to_process = request.fields_to_anonymize
            if not fields_to_process:
                # Auto-detect fields based on classification
                classification = await self._get_latest_classification(request.data.get('data_id', 'unknown'))
                if classification:
                    fields_to_process = classification.pii_fields
            
            # Apply anonymization
            for field in fields_to_process:
                if field in anonymized_data:
                    original_value = anonymized_data[field]
                    anonymized_value = self._apply_anonymization(
                        original_value, 
                        request.anonymization_method
                    )
                    anonymized_data[field] = anonymized_value
                    anonymized_fields.append(field)
                    
                    anonymization_log.append({
                        "field": field,
                        "method": request.anonymization_method.value,
                        "original_length": len(str(original_value)),
                        "anonymized_length": len(str(anonymized_value))
                    })
            
            # Calculate privacy score
            privacy_score = self._calculate_privacy_score(anonymized_data, anonymized_fields)
            
            # Determine compliance status
            compliance_status = "compliant" if privacy_score >= 0.8 else "needs_review"
            
            # Store anonymization log
            await self._store_anonymization_log(
                request.data.get('data_id', 'unknown'),
                request.policy_id,
                anonymized_fields,
                request.anonymization_method,
                privacy_score,
                compliance_status
            )
            
            return AnonymizationResponse(
                anonymized_data=anonymized_data,
                anonymized_fields=anonymized_fields,
                privacy_score=privacy_score,
                compliance_status=compliance_status,
                anonymization_log=anonymization_log
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to anonymize data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _apply_anonymization(self, value: Any, method: AnonymizationMethod) -> str:
        """Apply anonymization method to a value"""
        str_value = str(value)
        
        if method == AnonymizationMethod.HASH:
            return hashlib.sha256(str_value.encode()).hexdigest()[:8]
        elif method == AnonymizationMethod.MASK:
            if len(str_value) <= 4:
                return "*" * len(str_value)
            return str_value[:2] + "*" * (len(str_value) - 4) + str_value[-2:]
        elif method == AnonymizationMethod.REDACT:
            return "[REDACTED]"
        elif method == AnonymizationMethod.PSEUDONYMIZE:
            return f"PSEUDO_{hashlib.md5(str_value.encode()).hexdigest()[:8]}"
        elif method == AnonymizationMethod.GENERALIZE:
            if isinstance(value, (int, float)):
                return f"RANGE_{int(value // 10) * 10}-{int(value // 10) * 10 + 9}"
            return "GENERALIZED"
        else:
            return str_value
    
    def _calculate_privacy_score(self, data: Dict[str, Any], anonymized_fields: List[str]) -> float:
        """Calculate privacy score for anonymized data"""
        total_fields = len(data)
        if total_fields == 0:
            return 1.0
        
        anonymized_count = len(anonymized_fields)
        base_score = anonymized_count / total_fields
        
        # Bonus for anonymizing sensitive fields
        sensitive_bonus = 0.0
        for field in anonymized_fields:
            if any(pattern.name in field.lower() for pattern in self.pii_patterns):
                sensitive_bonus += 0.1
        
        return min(1.0, base_score + sensitive_bonus)
    
    async def _get_latest_classification(self, data_id: str) -> Optional[DataClassification]:
        """Get latest classification for data"""
        try:
            async with self.conn_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM data_classifications 
                    WHERE data_id = $1 
                    ORDER BY classified_at DESC 
                    LIMIT 1
                """, data_id)
                
                if row:
                    return DataClassification(
                        data_id=row['data_id'],
                        data_type=DataType(row['data_type']),
                        privacy_level=PrivacyLevel(row['privacy_level']),
                        contains_pii=row['contains_pii'],
                        pii_fields=json.loads(row['pii_fields']) if row['pii_fields'] else [],
                        sensitivity_score=row['sensitivity_score'],
                        classification_reason=row['classification_reason'],
                        classified_at=row['classified_at']
                    )
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get classification: {e}")
            return None
    
    async def _store_anonymization_log(self, data_id: str, policy_id: str, 
                                     anonymized_fields: List[str], method: AnonymizationMethod,
                                     privacy_score: float, compliance_status: str):
        """Store anonymization log"""
        try:
            async with self.conn_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO anonymization_logs 
                    (data_id, policy_id, anonymized_fields, anonymization_method, 
                     privacy_score, compliance_status, anonymized_at)
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                """, (
                    data_id, policy_id, json.dumps(anonymized_fields),
                    method.value, privacy_score, compliance_status
                ))
                
    except Exception as e:
            logger.error(f"❌ Failed to store anonymization log: {e}")
    
    async def generate_compliance_report(self) -> ComplianceReport:
        """Generate compliance report"""
        try:
            async with self.conn_pool.acquire() as conn:
                # Get total datasets
                total_datasets = await conn.fetchval("SELECT COUNT(DISTINCT data_id) FROM data_classifications")
                
                # Get compliant datasets (privacy_score >= 0.8)
                compliant_datasets = await conn.fetchval("""
                    SELECT COUNT(DISTINCT data_id) FROM anonymization_logs 
                    WHERE privacy_score >= 0.8
                """)
                
                # Get violations
                violations = await conn.fetch("""
                    SELECT data_id, privacy_score, compliance_status, anonymized_at
                    FROM anonymization_logs 
                    WHERE privacy_score < 0.8
                    ORDER BY privacy_score ASC
                """)
                
                non_compliant_datasets = len(violations)
                compliance_percentage = (compliant_datasets / total_datasets * 100) if total_datasets > 0 else 0
                
                # Generate recommendations
                recommendations = []
                if compliance_percentage < 80:
                    recommendations.append("Increase anonymization coverage for sensitive fields")
                if non_compliant_datasets > 0:
                    recommendations.append("Review and improve anonymization methods for low-scoring datasets")
                if total_datasets == 0:
                    recommendations.append("Implement data classification and anonymization processes")
                
                return ComplianceReport(
                    report_id=f"COMPLIANCE_{int(datetime.now().timestamp())}",
                    total_datasets=total_datasets,
                    compliant_datasets=compliant_datasets,
                    non_compliant_datasets=non_compliant_datasets,
                    compliance_percentage=compliance_percentage,
                    violations=[dict(v) for v in violations],
                    recommendations=recommendations,
                    generated_at=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to generate compliance report: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
privacy_service = DataPrivacyService()

# FastAPI app
app = FastAPI(title="Data Privacy Service", version="1.0.0")

# Setup distributed tracing
setup_tracing("data-privacy", app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await privacy_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Data Privacy Service...")
    try:
        await privacy_service.close()
        logger.info("✅ Data Privacy Service shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# API Routes
@app.post("/classify")
async def classify_data(data: Dict[str, Any], data_id: str):
    """Classify data for privacy compliance"""
    classification = await privacy_service.classify_data(data, data_id)
    return classification

@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_data(request: AnonymizationRequest):
    """Anonymize data according to privacy policy"""
    return await privacy_service.anonymize_data(request)

@app.get("/compliance-report", response_model=ComplianceReport)
async def get_compliance_report():
    """Generate compliance report"""
    return await privacy_service.generate_compliance_report()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-privacy",
        "uptime_seconds": (datetime.now() - privacy_service.start_time).total_seconds(),
        "pii_patterns_loaded": len(privacy_service.pii_patterns),
        "privacy_policies_loaded": len(privacy_service.privacy_policies)
    }

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Graceful shutdown handler
class GracefulShutdown:
    """Handles graceful shutdown of the data-privacy service"""
    
    def __init__(self, app):
        self.app = app
        self.is_shutting_down = False
        self.shutdown_timeout = 30  # seconds
        self.pending_tasks = set()
        logger.info("GracefulShutdown handler initialized for data-privacy service.")
    
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.shutdown_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self.shutdown_handler)
        logger.info("Registered SIGTERM and SIGINT handlers for data-privacy service.")
        
        @self.app.on_event("shutdown")
        async def _on_shutdown():
            await self._perform_cleanup()
    
    async def shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self.is_shutting_down:
            logger.warning("Data-privacy service already in shutdown process, ignoring signal.")
            return
        
        self.is_shutting_down = True
        logger.info(f"Data-privacy service received signal {signum}, initiating graceful shutdown...")
        
        # Trigger FastAPI's shutdown event
        await self.app.shutdown()
    
    async def _perform_cleanup(self):
        """Perform actual cleanup tasks."""
        logger.info("Performing graceful shutdown cleanup for data-privacy service...")
        
        # Cancel any pending background tasks
        if self.pending_tasks:
            logger.info(f"Cancelling {len(self.pending_tasks)} pending background tasks...")
            for task in list(self.pending_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task {task.get_name()} cancelled.")
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}")
            self.pending_tasks.clear()
            logger.info("All pending tasks cancelled.")
        
        # Close database connections
        try:
            if hasattr(privacy_service, 'conn_pool') and privacy_service.conn_pool:
                await privacy_service.conn_pool.close()
                logger.info("Database connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
        
        logger.info("Data-privacy service graceful shutdown cleanup complete.")

# Initialize graceful shutdown
shutdown_handler = GracefulShutdown(app)
shutdown_handler.register_handlers()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await shutdown_handler._perform_cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)