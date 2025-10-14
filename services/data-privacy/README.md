# ML Security Data Privacy Service

## Service Architecture & Purpose

### Core Purpose
The Data Privacy Service is the **compliance and data protection engine** of the ML Security platform. It ensures data privacy compliance, implements data anonymization and classification, and provides comprehensive privacy controls for GDPR, CCPA, and other regulatory requirements.

### Why This Service Exists
- **Regulatory Compliance**: Ensures compliance with GDPR, CCPA, HIPAA, and other privacy regulations
- **Data Classification**: Automatically classifies data based on sensitivity and privacy requirements
- **Anonymization**: Provides multiple anonymization techniques to protect sensitive data
- **Privacy Monitoring**: Continuous monitoring of data privacy compliance across the platform
- **Audit Trail**: Comprehensive logging and reporting for privacy compliance audits

## Complete API Documentation for Frontend Development

### Base URL
```
http://data-privacy:8005
```


### Core Privacy Endpoints

#### `POST /classify`
**Purpose**: Classify data for privacy compliance
**Request Body**:
```typescript
interface ClassificationRequest {
  data: Record<string, any>;
  data_id: string;
  context?: string;
  user_id?: string;
  session_id?: string;
}
```

**Frontend Usage**:
```javascript
const classificationRequest = {
  data: {
    name: "John Doe",
    email: "john.doe@example.com",
    phone: "555-123-4567",
    ssn: "123-45-6789",
    address: "123 Main St, Anytown, USA"
  },
  data_id: "user_profile_123",
  context: "user_registration",
  user_id: "user_456",
  session_id: "session_789"
};

const response = await fetch('/classify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(classificationRequest)
});

const result = await response.json();
console.log(`Data classified as: ${result.privacy_level}`);
console.log(`PII fields detected: ${result.pii_fields.join(', ')}`);
```

**Response Model**:
```typescript
interface ClassificationResponse {
  data_id: string;
  data_type: 'pii' | 'business' | 'public' | 'restricted';
  privacy_level: 'public' | 'internal' | 'confidential' | 'restricted';
  contains_pii: boolean;
  pii_fields: string[];
  sensitivity_score: number;
  classification_reason: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  compliance_status: {
    gdpr: 'compliant' | 'non_compliant' | 'requires_review';
    ccpa: 'compliant' | 'non_compliant' | 'requires_review';
    hipaa: 'compliant' | 'non_compliant' | 'requires_review';
  };
  recommendations: string[];
  classified_at: string;
}
```

#### `POST /anonymize`
**Purpose**: Anonymize data according to privacy policy
**Request Body**:
```typescript
interface AnonymizationRequest {
  data: Record<string, any>;
  policy_id?: string;
  fields_to_anonymize?: string[];
  anonymization_method?: 'hash' | 'mask' | 'redact' | 'pseudonymize' | 'generalize';
  preserve_format?: boolean;
  custom_rules?: Record<string, any>;
}
```

**Frontend Usage**:
```javascript
const anonymizationRequest = {
  data: {
    name: "John Doe",
    email: "john.doe@example.com",
    phone: "555-123-4567",
    ssn: "123-45-6789"
  },
  policy_id: "default",
  fields_to_anonymize: ["name", "email", "ssn"],
  anonymization_method: "hash",
  preserve_format: true
};

const response = await fetch('/anonymize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(anonymizationRequest)
});

const result = await response.json();
console.log('Anonymized data:', result.anonymized_data);
console.log(`Privacy score: ${result.privacy_score}`);
```

**Response Model**:
```typescript
interface AnonymizationResponse {
  anonymized_data: Record<string, any>;
  anonymized_fields: string[];
  privacy_score: number;
  compliance_status: 'compliant' | 'non_compliant' | 'requires_review';
  anonymization_log: Array<{
    field: string;
    method: string;
    original_length: number;
    anonymized_length: number;
    success: boolean;
  }>;
  policy_applied: string;
  anonymized_at: string;
}
```

#### `GET /compliance-report`
**Purpose**: Generate comprehensive compliance report
**Query Parameters**:
- `time_range` (optional): Time range (1d, 7d, 30d, 90d) - default: 30d
- `framework` (optional): Compliance framework (gdpr, ccpa, hipaa, all) - default: all
- `format` (optional): Report format (json, pdf, csv) - default: json

**Frontend Usage**:
```javascript
// Get GDPR compliance report for last 30 days
const response = await fetch('/compliance-report?framework=gdpr&time_range=30d');
const gdprReport = await response.json();

// Get comprehensive compliance report
const response = await fetch('/compliance-report?framework=all&time_range=7d');
const fullReport = await response.json();

// Display compliance metrics
console.log(`GDPR Compliance: ${fullReport.gdpr.compliance_percentage}%`);
console.log(`CCPA Compliance: ${fullReport.ccpa.compliance_percentage}%`);
```

**Response Model**:
```typescript
interface ComplianceReport {
  report_id: string;
  generated_at: string;
  time_range: string;
  frameworks: {
    gdpr: {
      compliance_percentage: number;
      total_datasets: number;
      compliant_datasets: number;
      violations: Array<{
        type: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
        description: string;
        affected_data: string[];
        remediation: string;
      }>;
      recommendations: string[];
    };
    ccpa: {
      compliance_percentage: number;
      total_datasets: number;
      compliant_datasets: number;
      violations: Array<{
        type: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
        description: string;
        affected_data: string[];
        remediation: string;
      }>;
      recommendations: string[];
    };
    hipaa: {
      compliance_percentage: number;
      total_datasets: number;
      compliant_datasets: number;
      violations: Array<{
        type: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
        description: string;
        affected_data: string[];
        remediation: string;
      }>;
      recommendations: string[];
    };
  };
  overall_compliance: number;
  risk_assessment: {
    overall_risk: 'low' | 'medium' | 'high' | 'critical';
    risk_factors: string[];
    mitigation_strategies: string[];
  };
  data_inventory: {
    total_datasets: number;
    pii_datasets: number;
    anonymized_datasets: number;
    public_datasets: number;
  };
}
```

### Privacy Policy Management Endpoints

#### `POST /policies`
**Purpose**: Create or update privacy policy
**Request Body**:
```typescript
interface PrivacyPolicyRequest {
  policy_id: string;
  name: string;
  description: string;
  framework: 'gdpr' | 'ccpa' | 'hipaa' | 'custom';
  rules: Array<{
    field_pattern: string;
    data_type: 'pii' | 'financial' | 'health' | 'biometric' | 'other';
    privacy_level: 'public' | 'internal' | 'confidential' | 'restricted';
    anonymization_method: 'hash' | 'mask' | 'redact' | 'pseudonymize' | 'generalize';
    retention_days: number;
    consent_required: boolean;
  }>;
  default_anonymization: 'hash' | 'mask' | 'redact' | 'pseudonymize' | 'generalize';
  retention_policy: {
    default_retention_days: number;
    auto_delete: boolean;
    archive_after_days: number;
  };
  consent_management: {
    explicit_consent_required: boolean;
    consent_expiry_days: number;
    consent_withdrawal_allowed: boolean;
  };
}
```

**Frontend Usage**:
```javascript
const policyRequest = {
  policy_id: "custom_policy_v1",
  name: "Custom Privacy Policy",
  description: "Custom privacy policy for ML Security platform",
  framework: "gdpr",
  rules: [
    {
      field_pattern: "email",
      data_type: "pii",
      privacy_level: "confidential",
      anonymization_method: "hash",
      retention_days: 365,
      consent_required: true
    },
    {
      field_pattern: "phone",
      data_type: "pii",
      privacy_level: "confidential",
      anonymization_method: "mask",
      retention_days: 365,
      consent_required: true
    }
  ],
  default_anonymization: "hash",
  retention_policy: {
    default_retention_days: 365,
    auto_delete: true,
    archive_after_days: 2555
  },
  consent_management: {
    explicit_consent_required: true,
    consent_expiry_days: 365,
    consent_withdrawal_allowed: true
  }
};

const response = await fetch('/policies', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(policyRequest)
});
```

#### `GET /policies`
**Purpose**: List all privacy policies
**Query Parameters**:
- `framework` (optional): Filter by compliance framework
- `active` (optional): Filter by active status - default: true

**Frontend Usage**:
```javascript
// Get all active policies
const response = await fetch('/policies?active=true');
const policies = await response.json();

// Get GDPR policies only
const response = await fetch('/policies?framework=gdpr');
const gdprPolicies = await response.json();
```

**Response Model**:
```typescript
interface PoliciesResponse {
  policies: Array<{
    policy_id: string;
    name: string;
    description: string;
    framework: string;
    version: string;
    created_at: string;
    updated_at: string;
    active: boolean;
    rules_count: number;
    compliance_score: number;
  }>;
  total_count: number;
}
```

#### `GET /policies/{policy_id}`
**Purpose**: Get detailed information about a specific policy
**Path Parameters**:
- `policy_id`: Unique identifier for the policy

#### `PUT /policies/{policy_id}`
**Purpose**: Update an existing privacy policy
**Path Parameters**:
- `policy_id`: Unique identifier for the policy

#### `DELETE /policies/{policy_id}`
**Purpose**: Delete a privacy policy
**Path Parameters**:
- `policy_id`: Unique identifier for the policy

### Data Classification Endpoints

#### `GET /classification/history`
**Purpose**: Get classification history
**Query Parameters**:
- `data_id` (optional): Filter by specific data ID
- `privacy_level` (optional): Filter by privacy level
- `time_range` (optional): Time range - default: 30d
- `limit` (optional): Maximum number of records - default: 100

**Frontend Usage**:
```javascript
// Get classification history for last 7 days
const response = await fetch('/classification/history?time_range=7d&limit=50');
const history = await response.json();

// Get history for specific data
const response = await fetch('/classification/history?data_id=user_profile_123');
const dataHistory = await response.json();
```

**Response Model**:
```typescript
interface ClassificationHistory {
  classifications: Array<{
    data_id: string;
    data_type: string;
    privacy_level: string;
    sensitivity_score: number;
    pii_fields: string[];
    classified_at: string;
    user_id?: string;
    session_id?: string;
  }>;
  total_count: number;
  time_range: string;
  summary: {
    total_classifications: number;
    pii_percentage: number;
    high_risk_percentage: number;
    compliance_violations: number;
  };
}
```

#### `GET /classification/patterns`
**Purpose**: Get PII detection patterns and statistics
**Query Parameters**:
- `pattern_type` (optional): Filter by pattern type (email, phone, ssn, etc.)
- `time_range` (optional): Time range - default: 30d

**Frontend Usage**:
```javascript
// Get all PII patterns
const response = await fetch('/classification/patterns');
const patterns = await response.json();

// Get email pattern statistics
const response = await fetch('/classification/patterns?pattern_type=email&time_range=7d');
const emailPatterns = await response.json();
```

**Response Model**:
```typescript
interface PIIPatterns {
  patterns: Array<{
    pattern_name: string;
    pattern_type: string;
    regex: string;
    sensitivity: number;
    detection_count: number;
    false_positive_rate: number;
    accuracy: number;
    last_detected: string;
  }>;
  total_patterns: number;
  detection_statistics: {
    total_detections: number;
    unique_patterns: number;
    accuracy_rate: number;
    false_positive_rate: number;
  };
}
```

### Anonymization Endpoints

#### `GET /anonymization/methods`
**Purpose**: Get available anonymization methods and their descriptions

**Frontend Usage**:
```javascript
const response = await fetch('/anonymization/methods');
const methods = await response.json();

// Display available methods
methods.methods.forEach(method => {
  console.log(`${method.name}: ${method.description}`);
});
```

**Response Model**:
```typescript
interface AnonymizationMethods {
  methods: Array<{
    name: string;
    description: string;
    privacy_level: 'low' | 'medium' | 'high';
    reversibility: 'irreversible' | 'reversible' | 'partially_reversible';
    use_cases: string[];
    examples: Array<{
      original: string;
      anonymized: string;
    }>;
  }>;
  default_method: string;
  recommended_methods: Record<string, string>;
}
```

#### `POST /anonymization/preview`
**Purpose**: Preview anonymization results without applying them
**Request Body**:
```typescript
interface AnonymizationPreviewRequest {
  data: Record<string, any>;
  fields_to_anonymize: string[];
  anonymization_method: string;
  preserve_format?: boolean;
}
```

**Frontend Usage**:
```javascript
const previewRequest = {
  data: {
    name: "John Doe",
    email: "john.doe@example.com",
    phone: "555-123-4567"
  },
  fields_to_anonymize: ["name", "email"],
  anonymization_method: "hash",
  preserve_format: true
};

const response = await fetch('/anonymization/preview', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(previewRequest)
});

const preview = await response.json();
console.log('Preview:', preview.preview_data);
```

#### `GET /anonymization/statistics`
**Purpose**: Get anonymization statistics and metrics
**Query Parameters**:
- `time_range` (optional): Time range - default: 30d
- `method` (optional): Filter by anonymization method

### Consent Management Endpoints

#### `POST /consent/record`
**Purpose**: Record user consent for data processing
**Request Body**:
```typescript
interface ConsentRecordRequest {
  user_id: string;
  data_types: string[];
  purposes: string[];
  consent_type: 'explicit' | 'implicit' | 'opt_in' | 'opt_out';
  consent_method: 'web_form' | 'api' | 'email' | 'phone' | 'other';
  ip_address?: string;
  user_agent?: string;
  metadata?: Record<string, any>;
}
```

**Frontend Usage**:
```javascript
const consentRequest = {
  user_id: "user_123",
  data_types: ["email", "phone", "name"],
  purposes: ["marketing", "analytics", "service_improvement"],
  consent_type: "explicit",
  consent_method: "web_form",
  ip_address: "192.168.1.1",
  user_agent: "Mozilla/5.0...",
  metadata: {
    form_version: "v2.1",
    language: "en-US"
  }
};

const response = await fetch('/consent/record', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(consentRequest)
});
```

#### `GET /consent/status/{user_id}`
**Purpose**: Get consent status for a specific user
**Path Parameters**:
- `user_id`: Unique identifier for the user

#### `POST /consent/withdraw`
**Purpose**: Withdraw user consent
**Request Body**:
```typescript
interface ConsentWithdrawalRequest {
  user_id: string;
  data_types?: string[];
  purposes?: string[];
  reason?: string;
}
```

#### `GET /consent/statistics`
**Purpose**: Get consent statistics and compliance metrics

### Health and Status Endpoints

#### `GET /health`
**Purpose**: Service health check

**Response Model**:
```typescript
interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  timestamp: string;
  uptime_seconds: number;
  dependencies: {
    database: boolean;
    redis: boolean;
    pattern_engine: boolean;
  };
  metrics: {
    total_classifications: number;
    total_anonymizations: number;
    active_policies: number;
    compliance_score: number;
  };
}
```

#### `GET /metrics`
**Purpose**: Prometheus metrics endpoint

### Frontend Integration Examples

#### Privacy Dashboard Component
```typescript
// React component for privacy dashboard
const PrivacyDashboard = () => {
  const [complianceReport, setComplianceReport] = useState(null);
  const [classificationHistory, setClassificationHistory] = useState(null);
  const [policies, setPolicies] = useState([]);

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        const [reportRes, historyRes, policiesRes] = await Promise.all([
          fetch('/compliance-report?framework=all&time_range=30d'),
          fetch('/classification/history?time_range=7d&limit=20'),
          fetch('/policies?active=true')
        ]);

        setComplianceReport(await reportRes.json());
        setClassificationHistory(await historyRes.json());
        setPolicies(await policiesRes.json());
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      }
    };

    loadDashboardData();
  }, []);

  return (
    <div className="privacy-dashboard">
      <ComplianceOverview report={complianceReport} />
      <ClassificationHistory history={classificationHistory} />
      <PoliciesList policies={policies} />
    </div>
  );
};
```

#### Data Classification Component
```typescript
// Component for data classification
const DataClassification = () => {
  const [data, setData] = useState('');
  const [classification, setClassification] = useState(null);
  const [loading, setLoading] = useState(false);

  const classifyData = async () => {
    if (!data.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: JSON.parse(data),
          data_id: `classification_${Date.now()}`,
          context: 'manual_classification'
        })
      });

      const result = await response.json();
      setClassification(result);
    } catch (error) {
      console.error('Classification failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="data-classification">
      <h3>Data Classification</h3>
      
      <textarea
        value={data}
        onChange={(e) => setData(e.target.value)}
        placeholder="Enter JSON data to classify..."
        rows={6}
      />
      
      <button onClick={classifyData} disabled={loading || !data}>
        {loading ? 'Classifying...' : 'Classify Data'}
      </button>
      
      {classification && (
        <ClassificationResult
          classification={classification}
          onAnonymize={() => {/* Handle anonymization */}}
        />
      )}
    </div>
  );
};
```

#### Anonymization Component
```typescript
// Component for data anonymization
const DataAnonymization = () => {
  const [data, setData] = useState('');
  const [anonymizedData, setAnonymizedData] = useState(null);
  const [method, setMethod] = useState('hash');
  const [fields, setFields] = useState([]);
  const [loading, setLoading] = useState(false);

  const anonymizeData = async () => {
    if (!data.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('/anonymize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: JSON.parse(data),
          anonymization_method: method,
          fields_to_anonymize: fields,
          preserve_format: true
        })
      });

      const result = await response.json();
      setAnonymizedData(result);
    } catch (error) {
      console.error('Anonymization failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const previewAnonymization = async () => {
    if (!data.trim()) return;

    try {
      const response = await fetch('/anonymization/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: JSON.parse(data),
          fields_to_anonymize: fields,
          anonymization_method: method
        })
      });

      const preview = await response.json();
      setAnonymizedData(preview);
    } catch (error) {
      console.error('Preview failed:', error);
    }
  };

  return (
    <div className="data-anonymization">
      <h3>Data Anonymization</h3>
      
      <textarea
        value={data}
        onChange={(e) => setData(e.target.value)}
        placeholder="Enter JSON data to anonymize..."
        rows={6}
      />
      
      <div className="anonymization-controls">
        <select value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="hash">Hash</option>
          <option value="mask">Mask</option>
          <option value="redact">Redact</option>
          <option value="pseudonymize">Pseudonymize</option>
          <option value="generalize">Generalize</option>
        </select>
        
        <FieldSelector
          data={data}
          selected={fields}
          onChange={setFields}
        />
      </div>
      
      <div className="action-buttons">
        <button onClick={previewAnonymization} disabled={!data}>
          Preview
        </button>
        <button onClick={anonymizeData} disabled={loading || !data}>
          {loading ? 'Anonymizing...' : 'Anonymize Data'}
        </button>
      </div>
      
      {anonymizedData && (
        <AnonymizationResult
          original={data}
          anonymized={anonymizedData.anonymized_data}
          log={anonymizedData.anonymization_log}
        />
      )}
    </div>
  );
};
```

## Architecture Deep Dive

### Service Components

#### 1. Data Classification Engine
**Purpose**: Automatically classifies data based on privacy sensitivity
**How it works**:
- Uses regex patterns and ML models to detect PII
- Implements sensitivity scoring algorithms
- Applies privacy level classification
- Provides detailed classification reasoning

#### 2. Anonymization Engine
**Purpose**: Applies various anonymization techniques to protect sensitive data
**How it works**:
- Implements multiple anonymization methods
- Supports field-level and record-level anonymization
- Provides configurable anonymization policies
- Maintains anonymization audit trails

#### 3. Privacy Policy Manager
**Purpose**: Manages privacy policies and compliance rules
**How it works**:
- Stores and manages privacy policies in database
- Applies policy-based data handling rules
- Supports multiple compliance frameworks
- Provides policy versioning and change tracking

#### 4. Compliance Monitor
**Purpose**: Monitors and reports on privacy compliance status
**How it works**:
- Tracks compliance metrics across all data processing
- Generates compliance reports and dashboards
- Identifies compliance violations and risks
- Provides remediation recommendations

### Data Flow Architecture

```
Data Input → Classification → Policy Check → Anonymization → Compliance Check → Output
     ↓            ↓              ↓             ↓              ↓            ↓
  Validation   PII Detection   Policy Match   Apply Method   Audit Log   Protected Data
     ↓            ↓              ↓             ↓              ↓            ↓
  Sanitization  Sensitivity     Rule Engine   Privacy Score  Compliance   Metadata
     ↓            ↓              ↓             ↓              ↓            ↓
  Classification  Privacy Level  Anonymization  Audit Trail   Reporting   Storage
```

## Configuration & Environment

### Environment Variables
```bash
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8005
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://mlflow:password@postgres:5432/ml_security_consolidated

# Privacy Configuration
DEFAULT_PRIVACY_LEVEL=internal
ANONYMIZATION_METHOD=hash
COMPLIANCE_FRAMEWORKS=gdpr,ccpa,hipaa
AUDIT_RETENTION_DAYS=2555  # 7 years
```

## Security & Compliance

### Data Protection
- **Encryption**: All data encrypted in transit and at rest
- **Access Control**: Role-based access to privacy functions
- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Minimization**: Only process necessary data

### Compliance Features
- **GDPR Compliance**: Right to be forgotten, data portability, consent management
- **CCPA Compliance**: Consumer rights, data disclosure, opt-out mechanisms
- **HIPAA Compliance**: Healthcare data protection and privacy
- **SOC 2**: Security and availability controls

## Performance Characteristics

### Scalability
- **Horizontal Scaling**: Stateless design supports multiple instances
- **Pattern Matching**: Optimized regex compilation and caching
- **Database Performance**: Indexed queries and connection pooling
- **Caching Strategy**: Policy and pattern caching for performance

### Reliability
- **Fault Tolerance**: Graceful handling of classification failures
- **Data Consistency**: ACID compliance for critical operations
- **Backup & Recovery**: Automated backup and point-in-time recovery
- **Error Handling**: Comprehensive error handling and recovery

## Troubleshooting Guide

### Common Issues
1. **Classification Failures**: Check PII patterns and data format
2. **Anonymization Errors**: Verify anonymization methods and policies
3. **Compliance Violations**: Review privacy policies and data handling
4. **Performance Issues**: Check database performance and caching

### Debug Commands
```bash
# Check service health
curl http://localhost:8005/health

# Test data classification
curl -X POST http://localhost:8005/classify \
  -H "Content-Type: application/json" \
  -d '{"data": {"email": "test@example.com"}, "data_id": "test_123"}'

# Generate compliance report
curl http://localhost:8005/compliance-report
```

## Future Enhancements

### Planned Features
- **ML-based Classification**: Machine learning models for PII detection
- **Differential Privacy**: Advanced privacy-preserving techniques
- **Consent Management**: User consent tracking and management
- **Privacy Impact Assessment**: Automated privacy impact analysis
- **Cross-border Compliance**: Multi-jurisdiction privacy compliance

### Research Areas
- **Federated Privacy**: Privacy-preserving distributed processing
- **Homomorphic Encryption**: Computation on encrypted data
- **Privacy-preserving Analytics**: Analytics without data exposure
- **Automated Compliance**: AI-driven compliance monitoring and reporting