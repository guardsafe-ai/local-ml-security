// =============================================================================
// COMPREHENSIVE TYPE DEFINITIONS FOR ML SECURITY DASHBOARD
// =============================================================================

// =============================================================================
// SYSTEM & HEALTH TYPES
// =============================================================================

export interface ServiceHealth {
  name: string;
  status: 'healthy' | 'unhealthy' | 'degraded' | 'unknown';
  response_time?: number;
  last_check: string;
  details?: {
    service: string;
    status: string;
    timestamp: string;
    uptime_seconds?: number;
    dependencies?: Record<string, boolean>;
    running?: boolean;
    total_tests?: number;
    models_loaded?: number;
    active_jobs?: number;
    error?: string;
  };
}

export interface SystemStatus {
  overall_status: 'healthy' | 'unhealthy' | 'degraded';
  services: ServiceHealth[];
  last_updated: string;
  system_metrics: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_usage: number;
  };
}

export interface DashboardMetrics {
  total_models: number;
  loaded_models: number;
  active_training_jobs: number;
  completed_training_jobs: number;
  total_red_team_tests: number;
  detection_rate: number;
  system_health: 'healthy' | 'unhealthy' | 'degraded';
  last_updated: string;
}

// =============================================================================
// MODEL MANAGEMENT TYPES
// =============================================================================

export interface ModelInfo {
  name: string;
  type: 'pretrained' | 'trained';
  version?: string;
  source: 'Hugging Face' | 'MLflow' | 'Local';
  loaded: boolean;
  size_gb?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  last_trained?: string;
  description?: string;
  tags?: string[];
  status: 'available' | 'loading' | 'loaded' | 'error';
  loading_progress?: number;
  error_message?: string;
}

export interface ModelRegistry {
  model_registry: {
    [key: string]: {
      latest: string;
      best: string;
      versions: string[];
      stages: {
        [stage: string]: string;
      };
    };
  };
}

export interface ModelPrediction {
  text: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_predictions?: Record<string, any>;
  ensemble_used: boolean;
  processing_time_ms: number;
  timestamp: string;
  from_cache?: boolean;
  model_name?: string;
}

export interface ModelVersion {
  version: string;
  stage: string;
  created_at: string;
  description?: string;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
}

// =============================================================================
// TRAINING TYPES
// =============================================================================

export interface TrainingJob {
  job_id: string;
  model_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
  training_data_path?: string;
  learning_rate?: number;
  batch_size?: number;
  num_epochs?: number;
  max_length?: number;
  config?: TrainingConfig;
  metrics?: TrainingMetrics;
  logs?: string[];
  error?: string;
  error_message?: string;
  result?: {
    status: string;
    metrics?: {
      epoch?: number;
      eval_loss?: number;
      eval_runtime?: number;
      eval_steps_per_second?: number;
      eval_samples_per_second?: number;
      [key: string]: any;
    };
    storage?: string;
    model_version?: string;
    training_time?: number;
    mlflow_model_uri?: string;
  };
}

export interface TrainingConfig {
  model_name: string;
  training_data_path: string;
  hyperparameters: {
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
    optimizer?: string;
    scheduler?: string;
  };
  validation_split?: number;
  test_split?: number;
  early_stopping?: boolean;
  patience?: number;
  metric_for_best_model?: string;
  // Additional fields from API response
  batch_size?: number;
  learning_rate?: number;
  num_epochs?: number;
  max_length?: number;
  evaluation_strategy?: string;
  eval_steps?: number;
  save_steps?: number;
  warmup_steps?: number;
  weight_decay?: number;
  load_best_model_at_end?: boolean;
  greater_is_better?: boolean;
}

export interface TrainingMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  loss: number;
  val_accuracy: number;
  val_loss: number;
  training_time_seconds: number;
  dataset_size: number;
  epochs_completed: number;
}

export interface TrainingData {
  text: string;
  label: string;
  severity?: number;
  source?: string;
  timestamp?: string;
}

// =============================================================================
// RED TEAM TESTING TYPES
// =============================================================================

export interface AttackPattern {
  category: 'prompt_injection' | 'jailbreak' | 'system_extraction' | 'code_injection' | 'benign';
  pattern: string;
  severity: number;
  description: string;
  timestamp: string;
}

export interface AttackResult {
  attack: AttackPattern;
  model_results: Record<string, any>;
  detected: boolean;
  confidence: number;
  timestamp: string;
  test_status: 'PASS' | 'FAIL' | 'UNKNOWN';
  pass_fail: boolean;
  detection_success: boolean;
  vulnerability_found: boolean;
  security_risk: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  model_name?: string;
  model_type?: string;
  model_version?: string;
}

export interface RedTeamTestResult {
  test_id: string;
  model_name: string;
  model_type: string;
  model_version?: string;
  total_attacks: number;
  vulnerabilities_found: number;
  detection_rate: number;
  overall_status: 'PASS' | 'FAIL';
  pass_count: number;
  fail_count: number;
  pass_rate: number;
  test_summary: {
    total_tests: number;
    passed: number;
    failed: number;
    success_rate: number;
    overall_status: string;
  };
  security_risk_distribution: {
    CRITICAL: number;
    HIGH: number;
    MEDIUM: number;
    LOW: number;
  };
  risk_summary: {
    critical_vulnerabilities: number;
    high_risk_vulnerabilities: number;
    medium_risk_vulnerabilities: number;
    low_risk_vulnerabilities: number;
  };
  duration_ms: number;
  timestamp: string;
  results: AttackResult[];
}

export interface RedTeamStatus {
  running: boolean;
  total_tests: number;
  current_test?: string;
  last_test_time?: string;
  learning_enabled: boolean;
  vulnerability_count: number;
  retraining_threshold: number;
}

// =============================================================================
// ANALYTICS TYPES
// =============================================================================

export interface AnalyticsSummary {
  summary: Array<{
    model_name: string;
    model_type: string;
    total_tests: number;
    avg_detection_rate: number;
    avg_attacks: number;
    avg_vulnerabilities: number;
    last_test: string;
  }>;
}

export interface AnalyticsTrends {
  trends: Array<{
    test_date: string;
    model_name: string;
    model_type: string;
    avg_detection_rate: number;
    test_count: number;
  }>;
}

export interface ModelPerformance {
  model_name: string;
  model_type: string;
  model_version?: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  training_duration_seconds?: number;
  dataset_size?: number;
  created_at: string;
}

export interface SecurityMetrics {
  total_attacks: number;
  detected_attacks: number;
  detection_rate: number;
  false_positive_rate: number;
  false_negative_rate: number;
  attack_categories: Record<string, number>;
  risk_distribution: Record<string, number>;
  time_series_data: Array<{
    timestamp: string;
    attacks: number;
    detected: number;
    detection_rate: number;
  }>;
}

// =============================================================================
// MLFLOW TYPES
// =============================================================================

export interface MLflowExperiment {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: string;
  creation_time: number;
  last_update_time: number;
  tags: Record<string, string>;
}

export interface MLflowRun {
  run_id: string;
  experiment_id: string;
  status: 'RUNNING' | 'SCHEDULED' | 'FINISHED' | 'FAILED' | 'KILLED';
  start_time: number;
  end_time?: number;
  artifact_uri: string;
  lifecycle_stage: string;
  tags: Record<string, string>;
  metrics: Record<string, number>;
  params: Record<string, string>;
}

export interface MLflowModel {
  name: string;
  latest_versions: Array<{
    name: string;
    version: string;
    creation_timestamp: number;
    last_updated_timestamp: number;
    current_stage: string;
    description?: string;
    user_id?: string;
    source: string;
    run_id: string;
    status: string;
    status_message?: string;
  }>;
  creation_timestamp: number;
  last_updated_timestamp: number;
  description?: string;
  tags: Record<string, string>;
}

// =============================================================================
// BUSINESS METRICS TYPES
// =============================================================================

export interface BusinessMetrics {
  total_cost: number;
  cost_per_prediction: number;
  cost_per_training_hour: number;
  roi_percentage: number;
  resource_utilization: {
    cpu_usage: number;
    memory_usage: number;
    storage_usage: number;
    network_usage: number;
  };
  performance_metrics: {
    average_response_time: number;
    throughput_per_second: number;
    error_rate: number;
    availability_percentage: number;
  };
  cost_breakdown: {
    compute_costs: number;
    storage_costs: number;
    network_costs: number;
    licensing_costs: number;
  };
}

// =============================================================================
// DATA PRIVACY TYPES
// =============================================================================

export interface DataPrivacyStatus {
  gdpr_compliant: boolean;
  data_anonymization_enabled: boolean;
  audit_logging_enabled: boolean;
  consent_management_enabled: boolean;
  data_retention_policy: string;
  last_audit_date: string;
  compliance_score: number;
  violations: Array<{
    type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    description: string;
    timestamp: string;
    resolved: boolean;
  }>;
}

// =============================================================================
// MODEL CACHE TYPES
// =============================================================================

export interface ModelCacheStatus {
  total_models_cached: number;
  cache_hit_rate: number;
  cache_miss_rate: number;
  memory_usage_mb: number;
  max_memory_mb: number;
  eviction_policy: string;
  last_cleanup: string;
  models: Array<{
    name: string;
    loaded: boolean;
    loaded_at: string;
    type: 'pretrained' | 'trained';
    memory_usage_mb: number;
    last_accessed: string;
    access_count: number;
  }>;
}

export interface ModelCacheStats {
  total_predictions: number;
  cache_hits: number;
  cache_misses: number;
  model_loads: number;
  start_time: string;
  uptime_seconds: number;
  average_response_time_ms: number;
  error_count: number;
}

// =============================================================================
// MONITORING TYPES
// =============================================================================

export interface PrometheusMetrics {
  metrics: Array<{
    name: string;
    type: string;
    help: string;
    samples: Array<{
      labels: Record<string, string>;
      value: number;
      timestamp: number;
    }>;
  }>;
}

export interface GrafanaDashboard {
  id: string;
  title: string;
  description: string;
  tags: string[];
  panels: Array<{
    id: string;
    title: string;
    type: string;
    targets: Array<{
      expr: string;
      legendFormat: string;
    }>;
  }>;
}

export interface JaegerTrace {
  traceID: string;
  spans: Array<{
    spanID: string;
    operationName: string;
    startTime: number;
    duration: number;
    tags: Record<string, string>;
    logs: Array<{
      timestamp: number;
      fields: Record<string, string>;
    }>;
  }>;
  processes: Record<string, {
    serviceName: string;
    tags: Record<string, string>;
  }>;
}

// =============================================================================
// SYSTEM ADMINISTRATION TYPES
// =============================================================================

export interface SystemConfig {
  services: Record<string, {
    enabled: boolean;
    config: Record<string, any>;
  }>;
  database: {
    host: string;
    port: number;
    name: string;
    user: string;
  };
  redis: {
    host: string;
    port: number;
    password?: string;
  };
  mlflow: {
    tracking_uri: string;
    artifact_root: string;
  };
  minio: {
    endpoint: string;
    access_key: string;
    secret_key: string;
    bucket: string;
  };
}

export interface DatabaseStatus {
  connected: boolean;
  version: string;
  uptime_seconds: number;
  connections: {
    active: number;
    idle: number;
    total: number;
  };
  database_size_mb: number;
  tables: Array<{
    name: string;
    rows: number;
    size_mb: number;
  }>;
}

// =============================================================================
// UI COMPONENT TYPES
// =============================================================================

export interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
    fill?: boolean;
  }>;
}

export interface TableColumn {
  key: string;
  label: string;
  sortable?: boolean;
  render?: (value: any, row: any) => React.ReactNode;
}

export interface TableData {
  columns: TableColumn[];
  rows: any[];
  pagination?: {
    page: number;
    pageSize: number;
    total: number;
  };
  sorting?: {
    column: string;
    direction: 'asc' | 'desc';
  };
}

export interface NotificationData {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}

// =============================================================================
// API RESPONSE TYPES
// =============================================================================

export interface ApiResponse<T = any> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  timestamp: string;
  request_id?: string;
}

export interface PaginatedResponse<T = any> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
  status: 'success' | 'error';
  message?: string;
}

export interface ErrorResponse {
  error: string;
  details?: string;
  status_code: number;
  timestamp: string;
  request_id?: string;
}

// =============================================================================
// FORM TYPES
// =============================================================================

export interface ModelLoadForm {
  model_name: string;
  version?: string;
}

export interface TrainingForm {
  model_name: string;
  training_data_path: string;
  hyperparameters: {
    learning_rate: number;
    batch_size: number;
    epochs: number;
    optimizer: string;
  };
}

export interface RedTeamTestForm {
  model_name: string;
  test_count: number;
  attack_categories?: string[];
  test_texts?: string[];
}

export interface SystemConfigForm {
  services: Record<string, any>;
  database: any;
  redis: any;
  mlflow: any;
  minio: any;
}

// =============================================================================
// UTILITY TYPES
// =============================================================================

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export type SortDirection = 'asc' | 'desc';

export type FilterOperator = 'equals' | 'contains' | 'startsWith' | 'endsWith' | 'greaterThan' | 'lessThan';

export interface Filter {
  column: string;
  operator: FilterOperator;
  value: any;
}

export interface Sort {
  column: string;
  direction: SortDirection;
}

export interface Pagination {
  page: number;
  pageSize: number;
}

export interface SearchParams {
  query?: string;
  filters?: Filter[];
  sort?: Sort;
  pagination?: Pagination;
}

