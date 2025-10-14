# Data Privacy Service - API Reference

## Base URL
```
http://data-privacy:8005
```

## Authentication
All endpoints require authentication headers:
```javascript
headers: {
  'Authorization': 'Bearer <token>',
  'Content-Type': 'application/json'
}
```

## Endpoints

### Health & Status

#### `GET /`
**Purpose**: Root endpoint with service status

**Response**:
```typescript
interface RootResponse {
  service: string;
  version: string;
  status: string;
  description: string;
  timestamp: string;
  uptime_seconds: number;
}
```

**Example Response**:
```json
{
  "service": "data-privacy",
  "version": "1.0.0",
  "status": "running",
  "description": "Data privacy and compliance service",
  "timestamp": "2025-01-09T10:30:00Z",
  "uptime_seconds": 3600
}
```

#### `GET /health`
**Purpose**: Comprehensive health check with privacy metrics

**Response**:
```typescript
interface HealthResponse {
  status: "healthy" | "unhealthy" | "degraded";
  service: string;
  timestamp: string;
  uptime_seconds: number;
  dependencies: {
    postgres: boolean;
    redis: boolean;
  };
  privacy_metrics: {
    total_pii_records: number;
    anonymized_records: number;
    compliance_rate: number;
  };
}
```

**Frontend Integration**:
```javascript
const checkHealth = async () => {
  try {
    const response = await fetch('/health');
    const health = await response.json();
    
    if (health.status === 'healthy') {
      console.log('✅ Data Privacy service is healthy');
      console.log(`📊 PII Records: ${health.privacy_metrics.total_pii_records}`);
      console.log(`🔒 Compliance Rate: ${health.privacy_metrics.compliance_rate}%`);
    } else {
      console.error('❌ Data Privacy service is unhealthy:', health);
    }
  } catch (error) {
    console.error('Failed to check health:', error);
  }
};
```

#### `GET /metrics`
**Purpose**: Prometheus metrics endpoint

**Response**: Prometheus-formatted metrics
```
# HELP data_privacy_pii_detections_total Total number of PII detections
# TYPE data_privacy_pii_detections_total counter
data_privacy_pii_detections_total{pii_type="email"} 150

# HELP data_privacy_anonymizations_total Total number of anonymizations
# TYPE data_privacy_anonymizations_total counter
data_privacy_anonymizations_total{method="pseudonymization"} 75
```

### PII Detection & Classification

#### `POST /classify`
**Purpose**: Detect PII and classify data sensitivity

**Request Body**:
```typescript
interface ClassificationRequest {
  data: Record<string, any>;
  data_id: string;
}
```

**Response**:
```typescript
interface ClassificationResponse {
  data_id: string;
  contains_pii: boolean;
  pii_fields: string[];
  sensitivity_score: number;
  privacy_level: "public" | "internal" | "confidential" | "restricted" | "top_secret";
  data_type: "text" | "numeric" | "categorical" | "temporal" | "spatial" | "binary" | "structured" | "unstructured";
  risk_score: number;
  detected_pii: Array<{
    field: string;
    pii_type: string;
    confidence: number;
    value: string;
    masked_value: string;
  }>;
  recommendations: string[];
  timestamp: string;
}
```

**Frontend Usage**:
```javascript
const classifyData = async (data, dataId) => {
  try {
    const response = await fetch('/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: data,
        data_id: dataId
      })
    });
    
    const classification = await response.json();
    console.log(`Contains PII: ${classification.contains_pii}`);
    console.log(`Privacy Level: ${classification.privacy_level}`);
    console.log(`Risk Score: ${classification.risk_score}`);
    return classification;
  } catch (error) {
    console.error('Classification failed:', error);
  }
};
```

**Example Request**:
```json
{
  "data": {
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1-555-123-4567",
    "ssn": "123-45-6789",
    "address": "123 Main St, Anytown, USA"
  },
  "data_id": "user_profile_001"
}
```

**Example Response**:
```json
{
  "data_id": "user_profile_001",
  "contains_pii": true,
  "pii_fields": ["name", "email", "phone", "ssn", "address"],
  "sensitivity_score": 0.95,
  "privacy_level": "confidential",
  "data_type": "structured",
  "risk_score": 0.85,
  "detected_pii": [
    {
      "field": "name",
      "pii_type": "name",
      "confidence": 0.98,
      "value": "John Doe",
      "masked_value": "J*** D***"
    },
    {
      "field": "email",
      "pii_type": "email",
      "confidence": 0.99,
      "value": "john.doe@example.com",
      "masked_value": "j***@example.com"
    }
  ],
  "recommendations": [
    "Apply pseudonymization to name field",
    "Use masking for email address",
    "Consider data minimization for phone number"
  ],
  "timestamp": "2025-01-09T10:30:00Z"
}
```

### Data Anonymization

#### `POST /anonymize`
**Purpose**: Anonymize sensitive data

**Request Body**:
```typescript
interface AnonymizationRequest {
  data: Record<string, any>;
  anonymization_method: "masking" | "pseudonymization" | "generalization" | "suppression" | "perturbation" | "swapping" | "synthetic_data" | "differential_privacy" | "k_anonymity" | "l_diversity" | "t_closeness";
  privacy_level: "public" | "internal" | "confidential" | "restricted" | "top_secret";
  preserve_utility?: boolean;
  custom_rules?: Record<string, any>;
}
```

**Response**:
```typescript
interface AnonymizationResponse {
  original_data: Record<string, any>;
  anonymized_data: Record<string, any>;
  anonymization_method: string;
  privacy_level: string;
  utility_score: number;
  risk_score: number;
  anonymization_details: Array<{
    field: string;
    original_value: any;
    anonymized_value: any;
    method_used: string;
    confidence: number;
  }>;
  quality_metrics: {
    data_utility: number;
    privacy_protection: number;
    re_identification_risk: number;
    information_loss: number;
  };
  recommendations: string[];
  timestamp: string;
}
```

**Frontend Usage**:
```javascript
const anonymizeData = async (data, method, privacyLevel) => {
  try {
    const response = await fetch('/anonymize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: data,
        anonymization_method: method,
        privacy_level: privacyLevel,
        preserve_utility: true
      })
    });
    
    const result = await response.json();
    console.log(`Anonymized data: ${JSON.stringify(result.anonymized_data)}`);
    console.log(`Utility score: ${result.utility_score}`);
    console.log(`Risk score: ${result.risk_score}`);
    return result;
  } catch (error) {
    console.error('Anonymization failed:', error);
  }
};
```

#### `POST /anonymize/batch`
**Purpose**: Batch anonymization of multiple data records

**Request Body**:
```typescript
interface BatchAnonymizationRequest {
  data_batch: Array<Record<string, any>>;
  anonymization_method: string;
  privacy_level: string;
  preserve_utility?: boolean;
}
```

**Response**:
```typescript
interface BatchAnonymizationResponse {
  original_batch: Array<Record<string, any>>;
  anonymized_batch: Array<Record<string, any>>;
  anonymization_method: string;
  privacy_level: string;
  batch_utility_score: number;
  batch_risk_score: number;
  individual_results: Array<{
    record_id: number;
    utility_score: number;
    risk_score: number;
    anonymization_details: Array<{
      field: string;
      original_value: any;
      anonymized_value: any;
      method_used: string;
    }>;
  }>;
  quality_metrics: {
    average_utility: number;
    average_privacy_protection: number;
    max_re_identification_risk: number;
    information_loss: number;
  };
  timestamp: string;
}
```

### Compliance Reporting

#### `GET /compliance-report`
**Purpose**: Generate comprehensive compliance report

**Query Parameters**:
- `regulation` (optional): Specific regulation (GDPR, CCPA, HIPAA) - default: all
- `time_range` (optional): Time range for report (7d, 30d, 90d, 1y) - default: 30d
- `format` (optional): Report format (json, pdf, csv) - default: json

**Response**:
```typescript
interface ComplianceReport {
  regulation: string;
  time_range: string;
  compliance_status: "compliant" | "non_compliant" | "needs_review";
  compliance_score: number;
  pii_records_count: number;
  anonymized_records_count: number;
  compliance_breakdown: {
    data_minimization: number;
    purpose_limitation: number;
    storage_limitation: number;
    accuracy: number;
    security: number;
    transparency: number;
    user_rights: number;
  };
  violations: Array<{
    type: string;
    severity: "low" | "medium" | "high" | "critical";
    description: string;
    affected_records: number;
    recommendation: string;
  }>;
  recommendations: string[];
  audit_trail: Array<{
    timestamp: string;
    action: string;
    user_id: string;
    details: string;
  }>;
  generated_at: string;
}
```

**Frontend Usage**:
```javascript
const getComplianceReport = async (regulation = 'GDPR', timeRange = '30d') => {
  try {
    const response = await fetch(
      `/compliance-report?regulation=${regulation}&time_range=${timeRange}`
    );
    const report = await response.json();
    
    console.log(`Compliance Status: ${report.compliance_status}`);
    console.log(`Compliance Score: ${report.compliance_score}%`);
    console.log(`PII Records: ${report.pii_records_count}`);
    console.log(`Anonymized Records: ${report.anonymized_records_count}`);
    return report;
  } catch (error) {
    console.error('Failed to get compliance report:', error);
  }
};
```

#### `GET /compliance-status`
**Purpose**: Get current compliance status

**Response**:
```typescript
interface ComplianceStatusResponse {
  overall_status: "compliant" | "non_compliant" | "needs_review";
  compliance_score: number;
  regulations: Array<{
    name: string;
    status: "compliant" | "non_compliant" | "needs_review";
    score: number;
    last_checked: string;
  }>;
  critical_issues: number;
  warnings: number;
  last_audit: string;
  next_audit: string;
  timestamp: string;
}
```

### Privacy Policy Management

#### `GET /privacy-policy`
**Purpose**: Get current privacy policy

**Response**:
```typescript
interface PrivacyPolicyResponse {
  version: string;
  last_updated: string;
  data_retention_days: number;
  anonymization_required: boolean;
  consent_required: boolean;
  audit_logging: boolean;
  data_processing_purposes: string[];
  data_categories: string[];
  legal_basis: string[];
  data_subjects: string[];
  data_recipients: string[];
  data_transfers: {
    international_transfers: boolean;
    adequacy_decision: boolean;
    safeguards: string[];
  };
  user_rights: string[];
  contact_information: {
    dpo_email: string;
    dpo_phone: string;
    dpo_address: string;
  };
  timestamp: string;
}
```

#### `PUT /privacy-policy`
**Purpose**: Update privacy policy

**Request Body**:
```typescript
interface PrivacyPolicyUpdateRequest {
  data_retention_days?: number;
  anonymization_required?: boolean;
  consent_required?: boolean;
  audit_logging?: boolean;
  data_processing_purposes?: string[];
  data_categories?: string[];
  legal_basis?: string[];
  data_subjects?: string[];
  data_recipients?: string[];
  data_transfers?: {
    international_transfers?: boolean;
    adequacy_decision?: boolean;
    safeguards?: string[];
  };
  user_rights?: string[];
  contact_information?: {
    dpo_email?: string;
    dpo_phone?: string;
    dpo_address?: string;
  };
}
```

**Response**:
```typescript
interface PrivacyPolicyUpdateResponse {
  status: "success" | "error";
  message: string;
  updated_policy: PrivacyPolicyResponse;
  changes: string[];
  timestamp: string;
}
```

### Privacy Metrics

#### `GET /metrics`
**Purpose**: Get privacy-related metrics

**Query Parameters**:
- `time_range` (optional): Time range for metrics (24h, 7d, 30d) - default: 24h
- `metric_type` (optional): Type of metrics (pii, anonymization, compliance) - default: all

**Response**:
```typescript
interface PrivacyMetricsResponse {
  time_range: string;
  pii_metrics: {
    total_pii_records: number;
    pii_detection_rate: number;
    false_positive_rate: number;
    pii_types_detected: Record<string, number>;
  };
  anonymization_metrics: {
    total_anonymized_records: number;
    anonymization_success_rate: number;
    average_utility_score: number;
    average_risk_score: number;
    anonymization_methods_used: Record<string, number>;
  };
  compliance_metrics: {
    compliance_rate: number;
    violation_count: number;
    audit_events: number;
    policy_updates: number;
  };
  quality_metrics: {
    data_quality_score: number;
    anonymization_quality_score: number;
    privacy_protection_score: number;
  };
  timestamp: string;
}
```

#### `GET /metrics/trends`
**Purpose**: Get privacy metrics trends

**Query Parameters**:
- `metric_name` (optional): Specific metric to analyze
- `time_range` (optional): Time range for trends (7d, 30d, 90d) - default: 30d
- `granularity` (optional): Trend granularity (1h, 1d, 1w) - default: 1d

**Response**:
```typescript
interface PrivacyTrendsResponse {
  metric_name: string;
  time_range: string;
  granularity: string;
  trends: Array<{
    timestamp: string;
    value: number;
    change_percent: number;
  }>;
  trend_analysis: {
    direction: "increasing" | "decreasing" | "stable";
    strength: number;
    correlation: number;
  };
  predictions: Array<{
    timestamp: string;
    predicted_value: number;
    confidence_interval: {
      lower: number;
      upper: number;
    };
  }>;
  timestamp: string;
}
```

### Data Quality Assessment

#### `POST /assess-quality`
**Purpose**: Assess data quality for privacy operations

**Request Body**:
```typescript
interface QualityAssessmentRequest {
  data: Record<string, any>;
  assessment_type: "privacy" | "anonymization" | "compliance" | "overall";
  criteria?: Record<string, any>;
}
```

**Response**:
```typescript
interface QualityAssessmentResponse {
  data_id: string;
  assessment_type: string;
  overall_score: number;
  dimension_scores: {
    completeness: number;
    accuracy: number;
    consistency: number;
    timeliness: number;
    validity: number;
    privacy_protection: number;
  };
  issues: Array<{
    dimension: string;
    severity: "low" | "medium" | "high";
    description: string;
    recommendation: string;
  }>;
  recommendations: string[];
  timestamp: string;
}
```

### Audit and Logging

#### `GET /audit-logs`
**Purpose**: Get audit logs for privacy operations

**Query Parameters**:
- `start_time` (optional): Start time for logs (ISO 8601)
- `end_time` (optional): End time for logs (ISO 8601)
- `operation_type` (optional): Filter by operation type
- `user_id` (optional): Filter by user ID
- `limit` (optional): Maximum number of logs to return - default: 100

**Response**:
```typescript
interface AuditLogsResponse {
  logs: Array<{
    id: string;
    timestamp: string;
    operation_type: string;
    user_id: string;
    resource: string;
    action: string;
    details: Record<string, any>;
    ip_address: string;
    user_agent: string;
    success: boolean;
    error_message?: string;
  }>;
  total_count: number;
  time_range: {
    start: string;
    end: string;
  };
  timestamp: string;
}
```

#### `POST /audit-log`
**Purpose**: Create custom audit log entry

**Request Body**:
```typescript
interface AuditLogRequest {
  operation_type: string;
  user_id: string;
  resource: string;
  action: string;
  details: Record<string, any>;
  ip_address?: string;
  user_agent?: string;
}
```

**Response**:
```typescript
interface AuditLogResponse {
  status: "success" | "error";
  message: string;
  log_id: string;
  timestamp: string;
}
```

## Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "error": true,
  "status_code": 400,
  "message": "Validation error",
  "details": [
    {
      "loc": ["body", "data"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "path": "/classify",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### 404 Not Found
```json
{
  "error": true,
  "status_code": 404,
  "message": "Data not found",
  "path": "/anonymize",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### 500 Internal Server Error
```json
{
  "error": true,
  "status_code": 500,
  "message": "Privacy processing failed",
  "path": "/classify",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

### Frontend Error Handling

```javascript
const handleApiError = (error, response) => {
  if (response?.status === 400) {
    console.error('Validation error:', response.data.details);
    // Show validation errors to user
  } else if (response?.status === 404) {
    console.error('Resource not found:', response.data.message);
    // Show not found message
  } else if (response?.status === 500) {
    console.error('Server error:', response.data.message);
    // Show generic error message
  } else {
    console.error('Unknown error:', error);
    // Show generic error message
  }
};
```

## Rate Limiting

The Data Privacy Service implements rate limiting to prevent abuse:

- **PII Detection**: 100 requests per minute per IP
- **Anonymization**: 50 requests per minute per IP
- **Compliance Reports**: 10 requests per minute per IP
- **Audit Logs**: 20 requests per minute per IP

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1641234567
```

## WebSocket Support

For real-time privacy monitoring, the service supports WebSocket connections:

```javascript
const ws = new WebSocket('ws://data-privacy:8005/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'pii_detected') {
    console.log(`PII detected in ${data.data_id}: ${data.pii_fields.join(', ')}`);
  } else if (data.type === 'anonymization_completed') {
    console.log(`Anonymization completed for ${data.data_id}`);
  }
};
```

## Request/Response Examples

### PII Detection
```bash
curl -X POST http://data-privacy:8005/classify \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "name": "John Doe",
      "email": "john.doe@example.com",
      "phone": "+1-555-123-4567"
    },
    "data_id": "user_001"
  }'
```

### Data Anonymization
```bash
curl -X POST http://data-privacy:8005/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "name": "John Doe",
      "email": "john.doe@example.com",
      "ssn": "123-45-6789"
    },
    "anonymization_method": "pseudonymization",
    "privacy_level": "confidential"
  }'
```

### Compliance Report
```bash
curl "http://data-privacy:8005/compliance-report?regulation=GDPR&time_range=30d"
```

### Privacy Metrics
```bash
curl "http://data-privacy:8005/metrics?time_range=24h&metric_type=pii"
```

---

**Data Privacy Service API** - Complete reference for all endpoints, request/response schemas, and integration examples for the ML Security Data Privacy Service.
