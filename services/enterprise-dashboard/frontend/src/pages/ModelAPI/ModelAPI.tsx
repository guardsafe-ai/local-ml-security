import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  Avatar,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  SmartToy,
  AutoAwesome,
  Memory,
  Speed,
  Refresh,
  Download,
  ViewModule,
  ViewList,
  TrendingUp,
  Assessment,
  Insights,
  CompareArrows,
  Security,
  Psychology,
  ModelTraining
} from '@mui/icons-material';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import ModelAPIDashboard from './ModelAPIDashboard';

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
      id={`model-api-tabpanel-${index}`}
      aria-labelledby={`model-api-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 0 }}>{children}</Box>}
    </div>
  );
}

const ModelAPI: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [viewMode, setViewMode] = useState<'dashboard' | 'advanced'>('dashboard');
  const [autoRefresh, setAutoRefresh] = useState(true);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleViewModeChange = () => {
    setViewMode(viewMode === 'dashboard' ? 'advanced' : 'dashboard');
  };

  const renderOverviewCards = () => (
    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={3}>
        <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom variant="h6">
                  Total Models
                </Typography>
                <Typography variant="h4" component="div" color="white">
                  4
                </Typography>
              </Box>
              <SmartToy sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom variant="h6">
                  Loaded Models
                </Typography>
                <Typography variant="h4" component="div" color="white">
                  2
                </Typography>
              </Box>
              <Memory sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom variant="h6">
                  Avg Confidence
                </Typography>
                <Typography variant="h4" component="div" color="white">
                  87.3%
                </Typography>
              </Box>
              <Speed sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom variant="h6">
                  Total Predictions
                </Typography>
                <Typography variant="h4" component="div" color="white">
                  1,247
                </Typography>
              </Box>
              <Psychology sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderQuickActions = () => (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Quick Actions
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              fullWidth
              onClick={() => window.location.reload()}
            >
              Refresh Models
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="outlined"
              startIcon={<Download />}
              fullWidth
              onClick={() => console.log('Export data')}
            >
              Export Report
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="outlined"
              startIcon={<AutoAwesome />}
              fullWidth
              onClick={() => console.log('Test prediction')}
            >
              Test Prediction
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="contained"
              startIcon={viewMode === 'dashboard' ? <ViewList /> : <ViewModule />}
              fullWidth
              onClick={handleViewModeChange}
            >
              {viewMode === 'dashboard' ? 'Advanced View' : 'Dashboard View'}
            </Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderBasicModelAPI = () => (
    <Box>
      {renderOverviewCards()}
      {renderQuickActions()}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  <SmartToy />
                </Avatar>
                <Box>
                  <Typography variant="h6">Model Management</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Load, unload, and manage ML models
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Manage PyTorch, Scikit-learn, and MLflow models with real-time loading and unloading.
              </Typography>
              <Chip label="4 models available" color="primary" variant="outlined" />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Avatar sx={{ bgcolor: 'success.main' }}>
                  <AutoAwesome />
                </Avatar>
                <Box>
                  <Typography variant="h6">Prediction Engine</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Real-time security threat detection
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Analyze text for security threats using ensemble models and batch processing.
              </Typography>
              <Chip label="87.3% accuracy" color="success" variant="outlined" />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Avatar sx={{ bgcolor: 'warning.main' }}>
                  <Memory />
                </Avatar>
                <Box>
                  <Typography variant="h6">Model Caching</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Redis-based model caching
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Intelligent caching system for improved performance and reduced latency.
              </Typography>
              <Chip label="2 models cached" color="warning" variant="outlined" />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Avatar sx={{ bgcolor: 'info.main' }}>
                  <Speed />
                </Avatar>
                <Box>
                  <Typography variant="h6">Performance Monitoring</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Real-time performance metrics
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Monitor model performance, response times, and resource utilization.
              </Typography>
              <Chip label="2.5ms avg response" color="info" variant="outlined" />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  return (
    <ErrorBoundary>
      <Box>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold', mb: 1 }}>
              Model API
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Unified API for security model inference, ensemble predictions, and model management
            </Typography>
          </Box>
          
          <Box display="flex" alignItems="center" gap={2}>
            <FormControlLabel
              control={
                <Switch 
                  checked={autoRefresh} 
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  color="primary"
                />
              }
              label="Auto Refresh"
            />
            
            <Tooltip title="Refresh Data">
              <IconButton onClick={() => window.location.reload()}>
                <Refresh />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Export Data">
              <IconButton onClick={() => console.log('Export')}>
                <Download />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* View Mode Toggle */}
        <Paper sx={{ mb: 3, p: 1 }}>
          <Box display="flex" justifyContent="center">
            <Tabs value={viewMode === 'dashboard' ? 0 : 1} onChange={(e, v) => setViewMode(v === 0 ? 'dashboard' : 'advanced')}>
              <Tab 
                label="Dashboard View" 
                icon={<ViewModule />} 
                iconPosition="start"
              />
              <Tab 
                label="Advanced Tools" 
                icon={<ViewList />} 
                iconPosition="start"
              />
            </Tabs>
          </Box>
        </Paper>

        {/* Content */}
        {viewMode === 'dashboard' ? (
          <Box>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="model api tabs" sx={{ mb: 3 }}>
              <Tab label="Overview" icon={<SmartToy />} />
              <Tab label="Model Management" icon={<ModelTraining />} />
              <Tab label="Prediction Engine" icon={<AutoAwesome />} />
              <Tab label="Cache Management" icon={<Memory />} />
              <Tab label="Performance" icon={<Speed />} />
              <Tab label="Analytics" icon={<Assessment />} />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              {renderBasicModelAPI()}
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Model Management
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Load, unload, and manage ML models with real-time status monitoring.
                  </Typography>
                </CardContent>
              </Card>
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Prediction Engine
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Real-time security threat detection using ensemble models and batch processing.
                  </Typography>
                </CardContent>
              </Card>
            </TabPanel>

            <TabPanel value={tabValue} index={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Cache Management
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Redis-based model caching for improved performance and reduced latency.
                  </Typography>
                </CardContent>
              </Card>
            </TabPanel>

            <TabPanel value={tabValue} index={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Performance Monitoring
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Monitor model performance, response times, and resource utilization.
                  </Typography>
                </CardContent>
              </Card>
            </TabPanel>

            <TabPanel value={tabValue} index={5}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Model Analytics
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Detailed analytics and insights for model performance and usage patterns.
                  </Typography>
                </CardContent>
              </Card>
            </TabPanel>
          </Box>
        ) : (
          <ModelAPIDashboard />
        )}
      </Box>
    </ErrorBoundary>
  );
};

export default ModelAPI;
