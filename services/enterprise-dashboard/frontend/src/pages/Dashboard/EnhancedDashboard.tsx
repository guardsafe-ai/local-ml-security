import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Security,
  Analytics,
  Business,
  Science,
  Monitor,
  Settings,
  Refresh,
  TrendingUp,
  TrendingDown,
  Speed,
  Memory,
  CloudSync,
  Timeline,
  Psychology,
  Assessment,
  Storage,
  Warning,
  CheckCircle,
  Error,
  Info,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { useWebSocket } from '../../services/websocketService';
import { apiService } from '../../services/apiService';
import RealTimeMetrics from '../../components/Dashboard/RealTimeMetrics';
import InsightsDashboard from '../../components/Dashboard/InsightsDashboard';
import ServiceHealthCard from '../../components/Dashboard/ServiceHealthCard';
import SystemMetricsGrid from '../../components/Dashboard/SystemMetricsGrid';
import ActivityTimeline from '../../components/Dashboard/ActivityTimeline';
import QuickActionsPanel from '../../components/Dashboard/QuickActionsPanel';
import LoadingSkeleton from '../../components/common/LoadingSkeleton/LoadingSkeleton';
import ErrorBoundary from '../../components/common/ErrorBoundary/ErrorBoundary';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const EnhancedDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as 'success' | 'error' | 'warning' | 'info' });
  const { isConnected, lastMessage } = useWebSocket();

  // Fetch dashboard metrics
  const { data: metrics, isLoading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useQuery({
    queryKey: ['dashboard-metrics'],
    queryFn: () => apiService.getDashboardMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch services health
  const { data: servicesHealth, isLoading: healthLoading, refetch: refetchHealth } = useQuery({
    queryKey: ['services-health'],
    queryFn: () => apiService.getServicesHealth(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch real-time metrics
  const { data: realTimeMetrics, isLoading: realTimeLoading, refetch: refetchRealTime } = useQuery({
    queryKey: ['real-time-metrics'],
    queryFn: () => apiService.getRealTimeMetrics(),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch recent activity
  const { data: recentActivity, isLoading: activityLoading, refetch: refetchActivity } = useQuery({
    queryKey: ['recent-activity'],
    queryFn: () => apiService.getRecentActivity(),
    refetchInterval: 15000, // Refresh every 15 seconds
  });

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'metrics_update':
          refetchMetrics();
          refetchRealTime();
          break;
        case 'alert':
          setSnackbar({
            open: true,
            message: lastMessage.data.message,
            severity: lastMessage.data.severity || 'info',
          });
          break;
        case 'system_status':
          refetchHealth();
          break;
        case 'activity_update':
          refetchActivity();
          break;
      }
    }
  }, [lastMessage, refetchMetrics, refetchRealTime, refetchHealth, refetchActivity]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    refetchMetrics();
    refetchHealth();
    refetchRealTime();
    refetchActivity();
  };

  const handleQuickAction = (actionId: string) => {
    console.log('Quick action triggered:', actionId);
    // Handle different quick actions
    switch (actionId) {
      case 'run_security_scan':
        setSnackbar({
          open: true,
          message: 'Security scan started',
          severity: 'info',
        });
        break;
      case 'start_training':
        setSnackbar({
          open: true,
          message: 'Training job initiated',
          severity: 'info',
        });
        break;
      case 'generate_report':
        setSnackbar({
          open: true,
          message: 'Report generation started',
          severity: 'info',
        });
        break;
      case 'refresh_data':
        handleRefresh();
        break;
      default:
        setSnackbar({
          open: true,
          message: `Action ${actionId} executed`,
          severity: 'info',
        });
    }
  };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
      case 'running':
      case 'active':
        return 'success';
      case 'warning':
      case 'degraded':
        return 'warning';
      case 'error':
      case 'failed':
      case 'down':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
      case 'running':
      case 'active':
        return <CheckCircle color="success" />;
      case 'warning':
      case 'degraded':
        return <Warning color="warning" />;
      case 'error':
      case 'failed':
      case 'down':
        return <Error color="error" />;
      default:
        return <Info color="info" />;
    }
  };

  const isLoading = metricsLoading || healthLoading || realTimeLoading || activityLoading;
  const hasError = metricsError || healthLoading === false && !servicesHealth;

  if (isLoading && !metrics) {
    return <LoadingSkeleton lines={8} />;
  }

  if (hasError) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load dashboard data. Please try again.
        <br />
        <button onClick={handleRefresh}>Retry</button>
      </Alert>
    );
  }

  const tabs = [
    { label: 'Overview', icon: <DashboardIcon />, value: 0 },
    { label: 'Real-time', icon: <Timeline />, value: 1 },
    { label: 'Analytics', icon: <Analytics />, value: 2 },
    { label: 'Security', icon: <Security />, value: 3 },
    { label: 'Performance', icon: <Speed />, value: 4 },
    { label: 'Business', icon: <Business />, value: 5 },
  ];

  return (
    <ErrorBoundary>
      <Box sx={{ flexGrow: 1, p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Enterprise ML Security Dashboard
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              icon={isConnected ? <CheckCircle /> : <Error />}
              label={isConnected ? 'Connected' : 'Disconnected'}
              color={isConnected ? 'success' : 'error'}
              variant="outlined"
            />
            <Tooltip title="Refresh Data">
              <IconButton onClick={handleRefresh} color="primary">
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="dashboard tabs">
            {tabs.map((tab) => (
              <Tab
                key={tab.value}
                icon={tab.icon}
                label={tab.label}
                iconPosition="start"
                sx={{ minHeight: 48 }}
              />
            ))}
          </Tabs>
        </Box>

        {/* Tab Panels */}
        <TabPanel value={activeTab} index={0}>
          {/* Overview Tab */}
          <Grid container spacing={3}>
            {/* System Metrics Grid */}
            <Grid item xs={12}>
              <SystemMetricsGrid
                metrics={metrics || {
                  total_models: 0,
                  active_jobs: 0,
                  total_attacks: 0,
                  detection_rate: 0,
                  system_health: 0
                }}
                onRefresh={handleRefresh}
                loading={isLoading}
              />
            </Grid>

            {/* Services Health Cards */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Services Health
              </Typography>
              <Grid container spacing={2}>
                {servicesHealth?.map((service: any, index: number) => (
                  <Grid item xs={12} sm={6} md={4} lg={2.4} key={index}>
                    <ServiceHealthCard
                      service={service}
                      onRefresh={handleRefresh}
                      loading={isLoading}
                    />
                  </Grid>
                )) || (
                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary">
                      No service health data available
                    </Typography>
                  </Grid>
                )}
              </Grid>
            </Grid>

            {/* Recent Activity and Quick Actions */}
            <Grid item xs={12} md={8}>
              <ActivityTimeline
                activities={recentActivity || []}
                onRefresh={handleRefresh}
                loading={isLoading}
                maxHeight={400}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <QuickActionsPanel
                onAction={handleQuickAction}
                loading={isLoading}
              />
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <RealTimeMetrics />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <InsightsDashboard />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <Typography variant="h6" gutterBottom>
            Security Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Security monitoring and threat detection features will be displayed here.
          </Typography>
        </TabPanel>

        <TabPanel value={activeTab} index={4}>
          <Typography variant="h6" gutterBottom>
            Performance Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            System performance metrics and optimization recommendations will be displayed here.
          </Typography>
        </TabPanel>

        <TabPanel value={activeTab} index={5}>
          <Typography variant="h6" gutterBottom>
            Business Metrics Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Business intelligence and cost analysis features will be displayed here.
          </Typography>
        </TabPanel>

        {/* Snackbar for notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
        >
          <Alert
            onClose={() => setSnackbar({ ...snackbar, open: false })}
            severity={snackbar.severity}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </ErrorBoundary>
  );
};

export default EnhancedDashboard;