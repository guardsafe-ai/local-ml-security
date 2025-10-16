import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';

// Types for Data Privacy Service
export interface DataSubject {
  subject_id: string;
  email?: string;
  created_at: string;
  last_accessed: string;
  data_categories: string[];
  retention_until: string;
  consent_given: boolean;
  consent_withdrawn: boolean;
}

export interface DataAnonymization {
  original_text: string;
  anonymized_text: string;
  anonymization_method: string;
  pii_detected: string[];
  confidence_score: number;
  anonymized_at: string;
}

export interface DataRetention {
  data_type: string;
  retention_period_days: number;
  auto_delete: boolean;
  last_cleanup: string;
  records_to_delete: int;
  compliance_status: string;
}

export interface AuditLog {
  log_id: string;
  timestamp: string;
  user_id?: string;
  action: string;
  resource: string;
  details: Record<string, any>;
  ip_address?: string;
  user_agent?: string;
  compliance_required: boolean;
}

export interface PrivacyCompliance {
  gdpr_compliant: boolean;
  data_subjects_count: number;
  consent_rate: number;
  data_retention_compliance: boolean;
  audit_logs_count: number;
  anonymization_rate: number;
  last_compliance_check: string;
  violations: string[];
}

// Custom hooks for Data Privacy Service
export const useDataPrivacy = (timeRange: number = 7) => {
  const [compliance, setCompliance] = useState<PrivacyCompliance | null>(null);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [dataSubjects, setDataSubjects] = useState<DataSubject[]>([]);
  const [retentionPolicies, setRetentionPolicies] = useState<DataRetention[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch all data privacy data in parallel
      const [
        complianceData,
        auditLogsData,
        dataSubjectsData,
        retentionPoliciesData
      ] = await Promise.all([
        apiService.getDataPrivacyCompliance(),
        apiService.getDataPrivacyAudit(),
        apiService.getDataSubjects(),
        apiService.getRetentionPolicies()
      ]);

      setCompliance(complianceData);
      setAuditLogs(auditLogsData.audit_logs || []);
      setDataSubjects(dataSubjectsData.data_subjects || []);
      setRetentionPolicies(retentionPoliciesData.retention_policies || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data privacy metrics');
      console.error('Data privacy error:', err);
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    compliance,
    auditLogs,
    dataSubjects,
    retentionPolicies,
    loading,
    error,
    refetch: fetchData
  };
};

// Hook for anonymization
export const useAnonymization = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const anonymizeText = useCallback(async (text: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.anonymizeData(text);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to anonymize text');
      console.error('Anonymization error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { anonymizeText, loading, error };
};

// Hook for data subject management
export const useDataSubjectManagement = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const registerDataSubject = useCallback(async (subjectData: {
    subject_id: string;
    email?: string;
    data_categories?: string[];
  }) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.registerDataSubject(subjectData);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to register data subject');
      console.error('Data subject registration error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const withdrawConsent = useCallback(async (subjectId: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.withdrawConsent(subjectId);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to withdraw consent');
      console.error('Consent withdrawal error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteDataSubject = useCallback(async (subjectId: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.deleteDataSubject(subjectId);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete data subject');
      console.error('Data subject deletion error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    registerDataSubject,
    withdrawConsent,
    deleteDataSubject,
    loading,
    error
  };
};

// Hook for compliance monitoring
export const useComplianceMonitoring = () => {
  const [compliance, setCompliance] = useState<PrivacyCompliance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCompliance = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getDataPrivacyCompliance();
      setCompliance(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch compliance status');
      console.error('Compliance monitoring error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCompliance();
  }, [fetchCompliance]);

  return { compliance, loading, error, refetch: fetchCompliance };
};

// Hook for audit logs
export const useAuditLogs = (filters?: {
  startDate?: string;
  endDate?: string;
  userId?: string;
  action?: string;
}) => {
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAuditLogs = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getDataPrivacyAudit(filters);
      setAuditLogs(response.audit_logs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch audit logs');
      console.error('Audit logs error:', err);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    fetchAuditLogs();
  }, [fetchAuditLogs]);

  return { auditLogs, loading, error, refetch: fetchAuditLogs };
};

// Hook for data cleanup
export const useDataCleanup = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cleanupExpiredData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.cleanupExpiredData();
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cleanup expired data');
      console.error('Data cleanup error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { cleanupExpiredData, loading, error };
};

// Hook for data privacy dashboard
export const useDataPrivacyDashboard = (timeRange: number = 7) => {
  const dataPrivacy = useDataPrivacy(timeRange);
  
  const isLoading = dataPrivacy.loading;
  const hasError = dataPrivacy.error;

  const refetchAll = useCallback(() => {
    dataPrivacy.refetch();
  }, [dataPrivacy.refetch]);

  return {
    ...dataPrivacy,
    loading: isLoading,
    error: hasError,
    refetch: refetchAll
  };
};
