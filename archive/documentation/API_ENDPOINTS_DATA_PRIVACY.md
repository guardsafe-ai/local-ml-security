# Data Privacy Service - Detailed Endpoints

**Base URL**: `http://localhost:8005`  
**Service**: Data Privacy Service  
**Purpose**: GDPR compliance and data protection

## Endpoints Overview

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| POST | `/anonymize` | Anonymize text data |
| POST | `/data-subjects` | Register data subject |
| POST | `/data-subjects/{subject_id}/withdraw-consent` | Withdraw consent |
| DELETE | `/data-subjects/{subject_id}` | Delete data subject |
| POST | `/cleanup` | Clean up expired data |
| GET | `/compliance` | Check privacy compliance |
| GET | `/audit-logs` | Get audit logs |
| GET | `/metrics` | Get Prometheus metrics |

---

## Detailed Endpoint Documentation

### 1. Health Check

#### `GET /`

**Purpose**: Basic health check endpoint

**Response**:
```json
{
  "status": "running",
  "service": "data-privacy"
}
```

**Example**:
```bash
curl http://localhost:8005/
```

---

### 2. Anonymize Data

#### `POST /anonymize`

**Purpose**: Anonymize text data by detecting and redacting PII

**Request Body**:
```json
{
  "text": "Contact John Doe at john@example.com or call 555-1234"
}
```

**Response**:
```json
{
  "original_text": "Contact John Doe at john@example.com or call 555-1234",
  "anonymized_text": "Contact [NAME_REDACTED] at [EMAIL_REDACTED] or call [PHONE_REDACTED]",
  "anonymization_method": "regex_replacement",
  "pii_detected": [
    {
      "type": "name",
      "value": "John Doe",
      "confidence": 0.95,
      "position": [8, 16]
    },
    {
      "type": "email",
      "value": "john@example.com",
      "confidence": 0.98,
      "position": [20, 35]
    },
    {
      "type": "phone",
      "value": "555-1234",
      "confidence": 0.92,
      "position": [45, 53]
    }
  ],
  "anonymization_quality": 0.95,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8005/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact John Doe at john@example.com or call 555-1234"}'
```

---

### 3. Register Data Subject

#### `POST /data-subjects`

**Purpose**: Register a new data subject for GDPR compliance

**Request Body**:
```json
{
  "subject_id": "user_123",
  "email": "user@example.com",
  "data_categories": ["personal_info", "usage_data", "preferences"],
  "consent_given": true,
  "retention_days": 365
}
```

**Response**:
```json
{
  "message": "Data subject registered successfully",
  "subject_id": "user_123",
  "data_subject": {
    "subject_id": "user_123",
    "email": "user@example.com",
    "created_at": "2024-01-01T00:00:00Z",
    "last_accessed": "2024-01-01T00:00:00Z",
    "data_categories": ["personal_info", "usage_data", "preferences"],
    "retention_until": "2024-12-31T00:00:00Z",
    "consent_given": true,
    "consent_withdrawn": false
  },
  "compliance_status": "compliant",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8005/data-subjects \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "user_123",
    "email": "user@example.com",
    "data_categories": ["personal_info", "usage_data"],
    "consent_given": true
  }'
```

---

### 4. Withdraw Consent

#### `POST /data-subjects/{subject_id}/withdraw-consent`

**Purpose**: Withdraw consent for a data subject

**Parameters**:
- `subject_id` (path): Data subject identifier

**Response**:
```json
{
  "message": "Consent withdrawn successfully",
  "subject_id": "user_123",
  "consent_status": "withdrawn",
  "data_retention": {
    "retention_until": "2024-01-08T00:00:00Z",
    "days_remaining": 7,
    "action_required": "Data will be deleted in 7 days"
  },
  "compliance_status": "pending_deletion",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8005/data-subjects/user_123/withdraw-consent
```

---

### 5. Delete Data Subject

#### `DELETE /data-subjects/{subject_id}`

**Purpose**: Delete all data for a data subject (right to be forgotten)

**Parameters**:
- `subject_id` (path): Data subject identifier

**Response**:
```json
{
  "message": "Data subject deleted successfully",
  "subject_id": "user_123",
  "deleted_data": {
    "personal_info": 15,
    "usage_data": 45,
    "preferences": 8,
    "total_records": 68
  },
  "deletion_verification": {
    "database_records": 0,
    "cache_entries": 0,
    "log_entries": 0,
    "verification_status": "complete"
  },
  "compliance_status": "deleted",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl -X DELETE http://localhost:8005/data-subjects/user_123
```

---

### 6. Clean Up Expired Data

#### `POST /cleanup`

**Purpose**: Clean up expired data based on retention policies

**Response**:
```json
{
  "message": "Data cleanup completed",
  "cleanup_summary": {
    "expired_subjects": 5,
    "records_deleted": 234,
    "data_categories_cleaned": {
      "personal_info": 45,
      "usage_data": 123,
      "preferences": 66
    },
    "storage_freed_mb": 12.5
  },
  "retention_policies": {
    "personal_info": 365,
    "usage_data": 90,
    "preferences": 180
  },
  "compliance_status": "compliant",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8005/cleanup
```

---

### 7. Check Privacy Compliance

#### `GET /compliance`

**Purpose**: Check overall privacy compliance status

**Response**:
```json
{
  "overall_compliance": "compliant",
  "compliance_score": 0.95,
  "gdpr_status": {
    "lawful_basis": "consent",
    "consent_rate": 0.92,
    "consent_withdrawal_rate": 0.05,
    "data_minimization": 0.98,
    "purpose_limitation": 0.96,
    "storage_limitation": 0.94,
    "accuracy": 0.97,
    "security": 0.99,
    "accountability": 0.93
  },
  "data_subjects": {
    "total_registered": 1250,
    "active_consent": 1150,
    "consent_withdrawn": 100,
    "deleted": 0,
    "pending_deletion": 15
  },
  "data_categories": {
    "personal_info": {
      "records": 1250,
      "retention_days": 365,
      "compliance_status": "compliant"
    },
    "usage_data": {
      "records": 4500,
      "retention_days": 90,
      "compliance_status": "compliant"
    },
    "preferences": {
      "records": 1250,
      "retention_days": 180,
      "compliance_status": "compliant"
    }
  },
  "audit_trail": {
    "total_events": 15420,
    "consent_events": 1250,
    "withdrawal_events": 100,
    "deletion_events": 0,
    "anonymization_events": 14070
  },
  "recommendations": [
    {
      "type": "consent_optimization",
      "priority": "low",
      "description": "Consider implementing granular consent options",
      "impact": "improved_user_experience"
    },
    {
      "type": "retention_optimization",
      "priority": "medium",
      "description": "Review usage_data retention period",
      "impact": "reduced_storage_costs"
    }
  ],
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8005/compliance
```

---

### 8. Get Audit Logs

#### `GET /audit-logs`

**Purpose**: Get audit logs for compliance tracking

**Parameters**:
- `start_date` (query, optional): Start date for logs (ISO format)
- `end_date` (query, optional): End date for logs (ISO format)
- `event_type` (query, optional): Filter by event type
- `subject_id` (query, optional): Filter by subject ID
- `limit` (query, optional): Maximum number of logs to return (default: 100)

**Response**:
```json
{
  "audit_logs": [
    {
      "log_id": "log_001",
      "timestamp": "2024-01-01T00:00:00Z",
      "user_id": "user_123",
      "action": "consent_given",
      "resource": "data_subject",
      "details": {
        "subject_id": "user_123",
        "data_categories": ["personal_info", "usage_data"],
        "consent_method": "explicit",
        "ip_address": "192.168.1.100"
      },
      "compliance_status": "compliant"
    },
    {
      "log_id": "log_002",
      "timestamp": "2024-01-01T00:01:00Z",
      "user_id": "user_123",
      "action": "data_anonymized",
      "resource": "text_data",
      "details": {
        "pii_types": ["name", "email"],
        "anonymization_method": "regex_replacement",
        "quality_score": 0.95
      },
      "compliance_status": "compliant"
    },
    {
      "log_id": "log_003",
      "timestamp": "2024-01-01T00:02:00Z",
      "user_id": "user_123",
      "action": "consent_withdrawn",
      "resource": "data_subject",
      "details": {
        "subject_id": "user_123",
        "withdrawal_method": "api",
        "retention_until": "2024-01-08T00:00:00Z"
      },
      "compliance_status": "pending_deletion"
    }
  ],
  "pagination": {
    "total_logs": 15420,
    "returned_logs": 3,
    "has_more": true,
    "next_offset": 3
  },
  "filters": {
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-01T23:59:59Z",
    "event_type": null,
    "subject_id": null,
    "limit": 100
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Examples**:
```bash
# Get all audit logs
curl http://localhost:8005/audit-logs

# Get logs for specific date range
curl "http://localhost:8005/audit-logs?start_date=2024-01-01T00:00:00Z&end_date=2024-01-01T23:59:59Z"

# Get logs for specific event type
curl "http://localhost:8005/audit-logs?event_type=consent_given"

# Get logs for specific subject
curl "http://localhost:8005/audit-logs?subject_id=user_123"

# Get limited number of logs
curl "http://localhost:8005/audit-logs?limit=50"
```

---

### 9. Get Prometheus Metrics

#### `GET /metrics`

**Purpose**: Get Prometheus-formatted metrics

**Response**:
```
# HELP data_privacy_data_subjects_total Total number of data subjects
# TYPE data_privacy_data_subjects_total gauge
data_privacy_data_subjects_total 1250

# HELP data_privacy_consent_rate Consent rate percentage
# TYPE data_privacy_consent_rate gauge
data_privacy_consent_rate 0.92

# HELP data_privacy_consent_withdrawal_rate Consent withdrawal rate percentage
# TYPE data_privacy_consent_withdrawal_rate gauge
data_privacy_consent_withdrawal_rate 0.05

# HELP data_privacy_compliance_score Overall compliance score
# TYPE data_privacy_compliance_score gauge
data_privacy_compliance_score 0.95

# HELP data_privacy_anonymization_events_total Total anonymization events
# TYPE data_privacy_anonymization_events_total counter
data_privacy_anonymization_events_total 14070

# HELP data_privacy_audit_events_total Total audit events
# TYPE data_privacy_audit_events_total counter
data_privacy_audit_events_total 15420

# HELP data_privacy_data_deletion_events_total Total data deletion events
# TYPE data_privacy_data_deletion_events_total counter
data_privacy_data_deletion_events_total 0

# HELP data_privacy_retention_policy_violations_total Total retention policy violations
# TYPE data_privacy_retention_policy_violations_total counter
data_privacy_retention_policy_violations_total 0
```

**Example**:
```bash
curl http://localhost:8005/metrics
```

---

## PII Detection Types

### Supported PII Types

| Type | Description | Example | Detection Method |
|------|-------------|---------|------------------|
| **Name** | Personal names | "John Doe", "Jane Smith" | NER + Regex |
| **Email** | Email addresses | "user@example.com" | Regex |
| **Phone** | Phone numbers | "555-1234", "+1-555-123-4567" | Regex |
| **SSN** | Social Security Numbers | "123-45-6789" | Regex |
| **Credit Card** | Credit card numbers | "4111-1111-1111-1111" | Regex + Luhn |
| **Address** | Physical addresses | "123 Main St, City, State" | NER + Regex |
| **IP Address** | IP addresses | "192.168.1.1" | Regex |
| **Date of Birth** | Birth dates | "01/01/1990", "January 1, 1990" | Regex |
| **Driver License** | Driver license numbers | "D123456789" | Regex |

### Anonymization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Regex Replacement** | Pattern-based replacement | Simple PII types |
| **NER (Named Entity Recognition)** | AI-based entity detection | Complex text analysis |
| **Hashing** | One-way hash generation | Data that needs to be linked |
| **Tokenization** | Replace with tokens | Structured data |
| **Masking** | Partial character replacement | Display purposes |

---

## GDPR Compliance Features

### 1. Lawful Basis

- **Consent**: Explicit user consent for data processing
- **Legitimate Interest**: Business necessity for processing
- **Contract**: Processing necessary for contract performance
- **Legal Obligation**: Processing required by law

### 2. Data Subject Rights

- **Right to Access**: View all personal data
- **Right to Rectification**: Correct inaccurate data
- **Right to Erasure**: Delete personal data (right to be forgotten)
- **Right to Portability**: Export personal data
- **Right to Restrict Processing**: Limit data processing
- **Right to Object**: Object to data processing

### 3. Data Protection Principles

- **Lawfulness**: Processing must have legal basis
- **Fairness**: Transparent and fair processing
- **Transparency**: Clear information about processing
- **Purpose Limitation**: Data used only for stated purposes
- **Data Minimization**: Collect only necessary data
- **Accuracy**: Keep data accurate and up-to-date
- **Storage Limitation**: Delete data when no longer needed
- **Security**: Protect data with appropriate measures
- **Accountability**: Demonstrate compliance

### 4. Consent Management

- **Explicit Consent**: Clear, unambiguous consent
- **Granular Consent**: Specific consent for different purposes
- **Withdrawal**: Easy consent withdrawal mechanism
- **Consent Records**: Detailed consent tracking
- **Consent Renewal**: Regular consent verification

---

## Error Responses

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input data |
| 404 | Not Found - Data subject not found |
| 409 | Conflict - Data subject already exists |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "status": "error",
  "error": "Data subject user_123 not found",
  "code": "SUBJECT_NOT_FOUND",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

---

## Usage Examples

### Python Client

```python
import requests
import json
from datetime import datetime, timedelta

# Anonymize text data
def anonymize_text(text):
    response = requests.post(
        "http://localhost:8005/anonymize",
        json={"text": text}
    )
    return response.json()

# Register data subject
def register_data_subject(subject_id, email, data_categories, consent=True):
    response = requests.post(
        "http://localhost:8005/data-subjects",
        json={
            "subject_id": subject_id,
            "email": email,
            "data_categories": data_categories,
            "consent_given": consent
        }
    )
    return response.json()

# Withdraw consent
def withdraw_consent(subject_id):
    response = requests.post(
        f"http://localhost:8005/data-subjects/{subject_id}/withdraw-consent"
    )
    return response.json()

# Delete data subject
def delete_data_subject(subject_id):
    response = requests.delete(
        f"http://localhost:8005/data-subjects/{subject_id}"
    )
    return response.json()

# Check compliance
def check_compliance():
    response = requests.get("http://localhost:8005/compliance")
    return response.json()

# Get audit logs
def get_audit_logs(start_date=None, end_date=None, event_type=None, subject_id=None):
    params = {}
    if start_date:
        params['start_date'] = start_date
    if end_date:
        params['end_date'] = end_date
    if event_type:
        params['event_type'] = event_type
    if subject_id:
        params['subject_id'] = subject_id
    
    response = requests.get("http://localhost:8005/audit-logs", params=params)
    return response.json()

# Example usage
# Anonymize text
text = "Contact John Doe at john@example.com or call 555-1234"
anonymized = anonymize_text(text)
print(f"Original: {anonymized['original_text']}")
print(f"Anonymized: {anonymized['anonymized_text']}")
print(f"PII detected: {len(anonymized['pii_detected'])} items")

# Register data subject
subject = register_data_subject(
    "user_123",
    "user@example.com",
    ["personal_info", "usage_data"],
    consent=True
)
print(f"Registered subject: {subject['subject_id']}")

# Check compliance
compliance = check_compliance()
print(f"Compliance score: {compliance['compliance_score']:.2%}")
print(f"GDPR status: {compliance['gdpr_status']['consent_rate']:.2%}")

# Get audit logs
logs = get_audit_logs(event_type="consent_given", limit=10)
print(f"Found {len(logs['audit_logs'])} consent events")
```

### JavaScript Client

```javascript
// Anonymize text data
async function anonymizeText(text) {
  const response = await fetch('http://localhost:8005/anonymize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text })
  });
  return await response.json();
}

// Register data subject
async function registerDataSubject(subjectId, email, dataCategories, consent = true) {
  const response = await fetch('http://localhost:8005/data-subjects', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      subject_id: subjectId,
      email: email,
      data_categories: dataCategories,
      consent_given: consent
    })
  });
  return await response.json();
}

// Withdraw consent
async function withdrawConsent(subjectId) {
  const response = await fetch(`http://localhost:8005/data-subjects/${subjectId}/withdraw-consent`, {
    method: 'POST'
  });
  return await response.json();
}

// Delete data subject
async function deleteDataSubject(subjectId) {
  const response = await fetch(`http://localhost:8005/data-subjects/${subjectId}`, {
    method: 'DELETE'
  });
  return await response.json();
}

// Check compliance
async function checkCompliance() {
  const response = await fetch('http://localhost:8005/compliance');
  return await response.json();
}

// Get audit logs
async function getAuditLogs(startDate = null, endDate = null, eventType = null, subjectId = null) {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  if (eventType) params.append('event_type', eventType);
  if (subjectId) params.append('subject_id', subjectId);
  
  const response = await fetch(`http://localhost:8005/audit-logs?${params}`);
  return await response.json();
}

// Example usage
// Anonymize text
const text = "Contact John Doe at john@example.com or call 555-1234";
anonymizeText(text).then(result => {
  console.log(`Original: ${result.original_text}`);
  console.log(`Anonymized: ${result.anonymized_text}`);
  console.log(`PII detected: ${result.pii_detected.length} items`);
});

// Register data subject
registerDataSubject(
  "user_123",
  "user@example.com",
  ["personal_info", "usage_data"],
  true
).then(result => {
  console.log(`Registered subject: ${result.subject_id}`);
});

// Check compliance
checkCompliance().then(compliance => {
  console.log(`Compliance score: ${(compliance.compliance_score * 100).toFixed(2)}%`);
  console.log(`GDPR status: ${(compliance.gdpr_status.consent_rate * 100).toFixed(2)}%`);
});

// Get audit logs
getAuditLogs(null, null, "consent_given", null).then(logs => {
  console.log(`Found ${logs.audit_logs.length} consent events`);
});
```

### cURL Examples

```bash
# Health check
curl http://localhost:8005/

# Anonymize text
curl -X POST http://localhost:8005/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact John Doe at john@example.com or call 555-1234"}'

# Register data subject
curl -X POST http://localhost:8005/data-subjects \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "user_123",
    "email": "user@example.com",
    "data_categories": ["personal_info", "usage_data"],
    "consent_given": true
  }'

# Withdraw consent
curl -X POST http://localhost:8005/data-subjects/user_123/withdraw-consent

# Delete data subject
curl -X DELETE http://localhost:8005/data-subjects/user_123

# Clean up expired data
curl -X POST http://localhost:8005/cleanup

# Check compliance
curl http://localhost:8005/compliance

# Get audit logs
curl http://localhost:8005/audit-logs

# Get audit logs with filters
curl "http://localhost:8005/audit-logs?event_type=consent_given&limit=50"

# Get Prometheus metrics
curl http://localhost:8005/metrics
```

---

## Integration Points

### Model API Integration

The Data Privacy Service integrates with the Model API Service to:
- Anonymize input data before processing
- Ensure PII is not exposed in predictions
- Maintain privacy compliance during inference

### Analytics Integration

Privacy data feeds into the Analytics Service for:
- Compliance reporting
- Privacy metrics tracking
- Audit trail analysis

### Monitoring Dashboard Integration

The Monitoring Dashboard displays privacy metrics:
- Compliance status
- Consent rates
- Audit logs
- Privacy controls

---

## Best Practices

### Data Anonymization

- Always anonymize data before processing
- Use appropriate anonymization methods
- Verify anonymization quality
- Maintain anonymization logs

### Consent Management

- Obtain explicit consent
- Provide granular consent options
- Make withdrawal easy
- Track consent changes

### Compliance Monitoring

- Regular compliance checks
- Monitor consent rates
- Track audit events
- Generate compliance reports

### Data Retention

- Implement retention policies
- Regular data cleanup
- Monitor retention compliance
- Document retention decisions

---

## Troubleshooting

### Common Issues

1. **PII detection not working**
   ```bash
   # Test anonymization
   curl -X POST http://localhost:8005/anonymize \
     -H "Content-Type: application/json" \
     -d '{"text": "Test with John Doe at john@example.com"}'
   
   # Check service logs
   docker-compose logs data-privacy
   ```

2. **Data subject not found**
   ```bash
   # Check if subject exists
   curl http://localhost:8005/compliance
   
   # Check audit logs
   curl "http://localhost:8005/audit-logs?subject_id=user_123"
   ```

3. **Compliance issues**
   ```bash
   # Check compliance status
   curl http://localhost:8005/compliance
   
   # Review audit logs
   curl http://localhost:8005/audit-logs
   ```

4. **Database connection issues**
   ```bash
   # Check PostgreSQL status
   docker-compose logs postgres
   
   # Check data-privacy service logs
   docker-compose logs data-privacy
   ```
