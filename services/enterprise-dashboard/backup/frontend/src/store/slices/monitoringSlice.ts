import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  responseTime: number;
  lastCheck: string;
  uptime: number;
  version?: string;
}

interface ResourceUsage {
  timestamp: string;
  cpu: number;
  memory: number;
  gpu: number;
  disk: number;
  network: number;
}

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  service: string;
  message: string;
  details?: any;
}

interface MonitoringState {
  services: ServiceStatus[];
  resourceUsage: ResourceUsage[];
  logs: LogEntry[];
  alerts: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'critical';
    title: string;
    message: string;
    timestamp: string;
    status: 'active' | 'acknowledged' | 'resolved';
    service?: string;
  }>;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  loading: boolean;
  error: string | null;
}

const initialState: MonitoringState = {
  services: [],
  resourceUsage: [],
  logs: [],
  alerts: [],
  connectionStatus: 'disconnected',
  loading: false,
  error: null,
};

const monitoringSlice = createSlice({
  name: 'monitoring',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setServices: (state, action: PayloadAction<ServiceStatus[]>) => {
      state.services = action.payload;
    },
    updateService: (state, action: PayloadAction<{ name: string; updates: Partial<ServiceStatus> }>) => {
      const { name, updates } = action.payload;
      const index = state.services.findIndex(service => service.name === name);
      if (index !== -1) {
        state.services[index] = { ...state.services[index], ...updates };
      }
    },
    setResourceUsage: (state, action: PayloadAction<ResourceUsage[]>) => {
      state.resourceUsage = action.payload;
    },
    addResourceUsage: (state, action: PayloadAction<ResourceUsage>) => {
      state.resourceUsage.push(action.payload);
      // Keep only last 100 entries
      if (state.resourceUsage.length > 100) {
        state.resourceUsage = state.resourceUsage.slice(-100);
      }
    },
    setLogs: (state, action: PayloadAction<LogEntry[]>) => {
      state.logs = action.payload;
    },
    addLog: (state, action: PayloadAction<LogEntry>) => {
      state.logs.unshift(action.payload);
      // Keep only last 1000 entries
      if (state.logs.length > 1000) {
        state.logs = state.logs.slice(0, 1000);
      }
    },
    clearLogs: (state) => {
      state.logs = [];
    },
    setAlerts: (state, action: PayloadAction<MonitoringState['alerts']>) => {
      state.alerts = action.payload;
    },
    addAlert: (state, action: PayloadAction<MonitoringState['alerts'][0]>) => {
      state.alerts.unshift(action.payload);
    },
    updateAlert: (state, action: PayloadAction<{ id: string; updates: Partial<MonitoringState['alerts'][0]> }>) => {
      const { id, updates } = action.payload;
      const index = state.alerts.findIndex(alert => alert.id === id);
      if (index !== -1) {
        state.alerts[index] = { ...state.alerts[index], ...updates };
      }
    },
    setConnectionStatus: (state, action: PayloadAction<MonitoringState['connectionStatus']>) => {
      state.connectionStatus = action.payload;
    },
  },
});

export const {
  setLoading,
  setError,
  setServices,
  updateService,
  setResourceUsage,
  addResourceUsage,
  setLogs,
  addLog,
  clearLogs,
  setAlerts,
  addAlert,
  updateAlert,
  setConnectionStatus,
} = monitoringSlice.actions;

export default monitoringSlice.reducer;
