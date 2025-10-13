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

  // Fetch comprehensive metrics
  const { data: metrics, isLoading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useQuery({
    queryKey: ['dashboard-metrics'],
    queryFn: () => apiService.getComprehensiveMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch real-time status
  const { data: realTimeStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['real-time-status'],
    queryFn: () => apiService.getRealTimeStatus(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'metrics_update':
          refetchMetrics();
          break;
        case 'alert':
          setSnackbar({
            open: true,
            message: lastMessage.data.message,
            severity: lastMessage.data.severity || 'info',
          });
          break;
        case 'system_status':
          // Handle system status updates
          break;
      }
    }
  }, [lastMessage, refetchMetrics]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    refetchMetrics();
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

  if (metricsLoading) {
    return <LoadingSkeleton lines={8} />;
  }

  if (metricsError) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load dashboard metrics. Please try again.
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
            {/* System Status Cards */}
            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Monitor color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">System Status</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusIcon(realTimeStatus?.system?.status || 'unknown')}
                    <Typography variant="body2">
                      {realTimeStatus?.system?.status || 'Unknown'}
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={realTimeStatus?.system?.uptime || 0}
                    sx={{ mt: 1 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    Uptime: {realTimeStatus?.system?.uptime || 0}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Security color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Security Status</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusIcon(realTimeStatus?.security?.status || 'unknown')}
                    <Typography variant="body2">
                      {realTimeStatus?.security?.status || 'Unknown'}
                    </Typography>
                  </Box>
                  <Typography variant="h4" color="primary" sx={{ mt: 1 }}>
                    {metrics?.security?.threats_blocked || 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Threats Blocked
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Analytics color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Model Performance</Typography>
                  </Box>
                  <Typography variant="h4" color="primary">
                    {metrics?.models?.average_accuracy || 0}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Average Accuracy
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUp color="success" sx={{ mr: 0.5 }} />
                    <Typography variant="body2" color="success.main">
                      +2.3% from last week
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Business color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Business Metrics</Typography>
                  </Box>
                  <Typography variant="h4" color="primary">
                    ${metrics?.business?.cost_savings || 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Cost Savings
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUp color="success" sx={{ mr: 0.5 }} />
                    <Typography variant="body2" color="success.main">
                      +15% this month
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Recent Activity */}
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Activity
                  </Typography>
                  <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                    {metrics?.recent_activity?.map((activity: any, index: number) => (
                      <Box key={index} sx={{ display: 'flex', alignItems: 'center', py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
                        <Box sx={{ mr: 2 }}>
                          {getStatusIcon(activity.status)}
                        </Box>
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="body2">{activity.message}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {new Date(activity.timestamp).toLocaleString()}
                          </Typography>
                        </Box>
                      </Box>
                    )) || (
                      <Typography variant="body2" color="text.secondary">
                        No recent activity
                      </Typography>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Quick Actions */}
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Quick Actions
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Chip
                      icon={<Science />}
                      label="Run Security Scan"
                      clickable
                      color="primary"
                      variant="outlined"
                    />
                    <Chip
                      icon={<Analytics />}
                      label="Generate Report"
                      clickable
                      color="secondary"
                      variant="outlined"
                    />
                    <Chip
                      icon={<Settings />}
                      label="Configure Alerts"
                      clickable
                      color="default"
                      variant="outlined"
                    />
                  </Box>
                </CardContent>
              </Card>
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