import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Layout Components
import NavigationMenu from './components/Layout/NavigationMenu/NavigationMenu';
import ErrorBoundary from './components/common/ErrorBoundary/ErrorBoundary';

// Dashboard Pages
import Dashboard from './pages/Dashboard/Dashboard';
import EnhancedDashboard from './pages/Dashboard/EnhancedDashboard';

// Model Management
import ModelRegistry from './pages/Models/ModelRegistry';
import ModelManagement from './pages/Models/ModelManagement';
import ModelTesting from './pages/Models/ModelTesting';
import ModelPerformance from './pages/Models/ModelPerformance';

// Training
import TrainingQueue from './pages/Training/TrainingQueue';
import DataManagement from './pages/Training/DataManagement';
import DataAugmentation from './pages/Training/DataAugmentation';
import TrainingConfig from './pages/Training/TrainingConfig';

// Analytics
import Analytics from './pages/Analytics/Analytics';
import PerformanceAnalytics from './pages/Analytics/PerformanceAnalytics';
import DriftDetection from './pages/Analytics/DriftDetection';
import AnalyticsAdvanced from './pages/Analytics/AnalyticsAdvanced';
import AnalyticsDashboard from './pages/Analytics/AnalyticsDashboard';

// Red Team
import RedTeamDashboard from './pages/RedTeam/RedTeamDashboard';
import AttackSimulation from './pages/RedTeam/AttackSimulation';
import VulnerabilityAssessment from './pages/RedTeam/VulnerabilityAssessment';
import ComplianceTesting from './pages/RedTeam/ComplianceTesting';

// Business Metrics
import BusinessMetrics from './pages/Business/BusinessMetrics';
import CostAnalysis from './pages/Business/CostAnalysis';
import ResourceUtilization from './pages/Business/ResourceUtilization';
import PerformanceKPIs from './pages/Business/PerformanceKPIs';
import BusinessMetricsDashboard from './pages/BusinessMetrics/BusinessMetricsDashboard';

// Data Privacy
import DataPrivacy from './pages/DataPrivacy/DataPrivacy';
import DataClassification from './pages/Privacy/DataClassification';
import PrivacyCompliance from './pages/Privacy/PrivacyCompliance';
import Anonymization from './pages/Privacy/Anonymization';
import DataPrivacyDashboard from './pages/DataPrivacy/DataPrivacyDashboard';

// MLflow
import MLflow from './pages/MLflow/MLflow';
import Experiments from './pages/MLflow/Experiments';
import MLflowRegistry from './pages/MLflow/MLflowRegistry';
import Artifacts from './pages/MLflow/Artifacts';
import RunComparison from './pages/MLflow/RunComparison';

// System
import ModelCache from './pages/System/ModelCache';
import HealthStatus from './pages/System/HealthStatus';

// Settings
import SecuritySettings from './pages/Settings/SecuritySettings';
import DataManagementSettings from './pages/Settings/DataManagement';
import UserManagement from './pages/Settings/UserManagement';

// Services
import { WebSocketProvider } from './services/websocketService';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00bcd4',
    },
    secondary: {
      main: '#ff4081',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

// Create query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

// Main App Component
const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <WebSocketProvider>
          <Router>
            <ErrorBoundary>
              <NavigationMenu />
              <Routes>
                {/* Dashboard Routes */}
                <Route path="/" element={<EnhancedDashboard />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/enhanced-dashboard" element={<EnhancedDashboard />} />

                {/* Model Management */}
                <Route path="/models" element={<ModelRegistry />} />
                <Route path="/models/registry" element={<ModelRegistry />} />
                <Route path="/models/management" element={<ModelManagement />} />
                <Route path="/models/testing" element={<ModelTesting />} />
                <Route path="/models/performance" element={<ModelPerformance />} />

                {/* Training */}
                <Route path="/training/queue" element={<TrainingQueue />} />
                <Route path="/training/data" element={<DataManagement />} />
                <Route path="/training/augmentation" element={<DataAugmentation />} />
                <Route path="/training/config" element={<TrainingConfig />} />

                {/* Analytics */}
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/analytics/performance" element={<PerformanceAnalytics />} />
                <Route path="/analytics/drift" element={<DriftDetection />} />
                <Route path="/analytics/advanced" element={<AnalyticsAdvanced />} />
                <Route path="/analytics/dashboard" element={<AnalyticsDashboard />} />

                {/* Red Team */}
                <Route path="/red-team" element={<RedTeamDashboard />} />
                <Route path="/red-team/simulation" element={<AttackSimulation />} />
                <Route path="/red-team/vulnerability" element={<VulnerabilityAssessment />} />
                <Route path="/red-team/compliance" element={<ComplianceTesting />} />

                {/* Business Metrics */}
                <Route path="/business" element={<BusinessMetrics />} />
                <Route path="/business/cost" element={<CostAnalysis />} />
                <Route path="/business/resources" element={<ResourceUtilization />} />
                <Route path="/business/kpis" element={<PerformanceKPIs />} />
                <Route path="/business-metrics" element={<BusinessMetricsDashboard />} />

                {/* Data Privacy */}
                <Route path="/privacy" element={<DataPrivacy />} />
                <Route path="/privacy/classification" element={<DataClassification />} />
                <Route path="/privacy/compliance" element={<PrivacyCompliance />} />
                <Route path="/privacy/anonymization" element={<Anonymization />} />
                <Route path="/data-privacy" element={<DataPrivacyDashboard />} />

                {/* MLflow */}
                <Route path="/mlflow" element={<MLflow />} />
                <Route path="/mlflow/experiments" element={<Experiments />} />
                <Route path="/mlflow/registry" element={<MLflowRegistry />} />
                <Route path="/mlflow/artifacts" element={<Artifacts />} />
                <Route path="/mlflow/comparison" element={<RunComparison />} />

                {/* System */}
                <Route path="/system/cache" element={<ModelCache />} />
                <Route path="/system/health" element={<HealthStatus />} />

                {/* Settings */}
                <Route path="/settings/security" element={<SecuritySettings />} />
                <Route path="/settings/data" element={<DataManagementSettings />} />
                <Route path="/settings/users" element={<UserManagement />} />
              </Routes>
            </ErrorBoundary>
          </Router>
        </WebSocketProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;