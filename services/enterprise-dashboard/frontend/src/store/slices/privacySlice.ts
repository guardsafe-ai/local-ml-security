import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface DataClassification {
  id: string;
  dataId: string;
  dataType: 'pii' | 'sensitive' | 'financial' | 'health' | 'business';
  privacyLevel: 'public' | 'internal' | 'confidential' | 'restricted';
  containsPii: boolean;
  piiFields: string[];
  sensitivityScore: number;
  classificationReason: string;
  classifiedAt: string;
}

interface PrivacyPolicy {
  id: string;
  name: string;
  description: string;
  privacyLevel: 'public' | 'internal' | 'confidential' | 'restricted';
  dataTypes: string[];
  retentionDays: number;
  anonymizationRequired: boolean;
  anonymizationMethod?: 'hash' | 'mask' | 'redact' | 'pseudonymize' | 'generalize';
  createdAt: string;
  updatedAt: string;
}

interface PrivacyState {
  classifications: DataClassification[];
  policies: PrivacyPolicy[];
  complianceScore: number;
  violations: Array<{
    id: string;
    type: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    timestamp: string;
    status: 'open' | 'acknowledged' | 'resolved';
  }>;
  loading: boolean;
  error: string | null;
}

const initialState: PrivacyState = {
  classifications: [],
  policies: [],
  complianceScore: 0,
  violations: [],
  loading: false,
  error: null,
};

const privacySlice = createSlice({
  name: 'privacy',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setClassifications: (state, action: PayloadAction<DataClassification[]>) => {
      state.classifications = action.payload;
    },
    addClassification: (state, action: PayloadAction<DataClassification>) => {
      state.classifications.unshift(action.payload);
    },
    setPolicies: (state, action: PayloadAction<PrivacyPolicy[]>) => {
      state.policies = action.payload;
    },
    addPolicy: (state, action: PayloadAction<PrivacyPolicy>) => {
      state.policies.unshift(action.payload);
    },
    updatePolicy: (state, action: PayloadAction<{ id: string; updates: Partial<PrivacyPolicy> }>) => {
      const { id, updates } = action.payload;
      const index = state.policies.findIndex(policy => policy.id === id);
      if (index !== -1) {
        state.policies[index] = { ...state.policies[index], ...updates };
      }
    },
    deletePolicy: (state, action: PayloadAction<string>) => {
      state.policies = state.policies.filter(policy => policy.id !== action.payload);
    },
    setComplianceScore: (state, action: PayloadAction<number>) => {
      state.complianceScore = action.payload;
    },
    setViolations: (state, action: PayloadAction<PrivacyState['violations']>) => {
      state.violations = action.payload;
    },
    addViolation: (state, action: PayloadAction<PrivacyState['violations'][0]>) => {
      state.violations.unshift(action.payload);
    },
    updateViolation: (state, action: PayloadAction<{ id: string; updates: Partial<PrivacyState['violations'][0]> }>) => {
      const { id, updates } = action.payload;
      const index = state.violations.findIndex(violation => violation.id === id);
      if (index !== -1) {
        state.violations[index] = { ...state.violations[index], ...updates };
      }
    },
  },
});

export const {
  setLoading,
  setError,
  setClassifications,
  addClassification,
  setPolicies,
  addPolicy,
  updatePolicy,
  deletePolicy,
  setComplianceScore,
  setViolations,
  addViolation,
  updateViolation,
} = privacySlice.actions;

export default privacySlice.reducer;
