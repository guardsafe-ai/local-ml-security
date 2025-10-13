import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  persistent?: boolean;
}

interface SystemSettings {
  theme: 'light' | 'dark';
  language: string;
  timezone: string;
  dateFormat: string;
  refreshInterval: number;
  notifications: {
    email: boolean;
    browser: boolean;
    sound: boolean;
  };
  dashboard: {
    layout: 'grid' | 'list';
    defaultView: string;
    autoRefresh: boolean;
  };
}

interface SystemState {
  notifications: Notification[];
  settings: SystemSettings;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  lastSync: string | null;
  loading: boolean;
  error: string | null;
}

const initialState: SystemState = {
  notifications: [],
  settings: {
    theme: 'dark',
    language: 'en',
    timezone: 'UTC',
    dateFormat: 'YYYY-MM-DD',
    refreshInterval: 30,
    notifications: {
      email: true,
      browser: true,
      sound: false,
    },
    dashboard: {
      layout: 'grid',
      defaultView: 'dashboard',
      autoRefresh: true,
    },
  },
  connectionStatus: 'disconnected',
  lastSync: null,
  loading: false,
  error: null,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp' | 'read'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        read: false,
      };
      state.notifications.unshift(notification);
      // Keep only last 100 notifications
      if (state.notifications.length > 100) {
        state.notifications = state.notifications.slice(0, 100);
      }
    },
    markNotificationAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    markAllNotificationsAsRead: (state) => {
      state.notifications.forEach(notification => {
        notification.read = true;
      });
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    updateSettings: (state, action: PayloadAction<Partial<SystemSettings>>) => {
      state.settings = { ...state.settings, ...action.payload };
    },
    setConnectionStatus: (state, action: PayloadAction<SystemState['connectionStatus']>) => {
      state.connectionStatus = action.payload;
    },
    setLastSync: (state, action: PayloadAction<string>) => {
      state.lastSync = action.payload;
    },
  },
});

export const {
  setLoading,
  setError,
  addNotification,
  markNotificationAsRead,
  markAllNotificationsAsRead,
  removeNotification,
  clearNotifications,
  updateSettings,
  setConnectionStatus,
  setLastSync,
} = systemSlice.actions;

export default systemSlice.reducer;
