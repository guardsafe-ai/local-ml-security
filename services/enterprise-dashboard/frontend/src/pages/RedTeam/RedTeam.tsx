import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Badge,
  Avatar,
  Skeleton,
  Fade,
  CircularProgress
} from '@mui/material';
import {
  Security,
  PlayArrow,
  Stop,
  Refresh,
  CheckCircle,
  Error,
  Dangerous,
  Warning,
  Info,
  Report,
  School
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import {
  useRedTeamStatus,
  useRedTeamResults,
  useRedTeamOperations,
  useModels
} from '../../hooks/useApi';
import { RedTeamTestResult, ModelInfo } from '../../types';

// =============================================================================
// RED TEAM TESTING PAGE
// =============================================================================

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
      id={`redteam-tabpanel-${index}`}
      aria-labelledby={`redteam-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const RedTeam: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedTest, setSelectedTest] = useState<RedTeamTestResult | null>(null);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [testConfig, setTestConfig] = useState({
    model_name: '',
    test_count: 10,
    attack_categories: ['prompt_injection', 'jailbreak', 'system_extraction', 'code_injection']
  });
  const [isTestRunning, setIsTestRunning] = useState(false);

  // API hooks - Manual loading only
  const { data: redTeamStatus, loading: statusLoading, refetch: refetchStatus, execute: executeStatus } = useRedTeamStatus();
  const { data: redTeamResults, loading: resultsLoading, refetch: refetchResults, execute: executeResults } = useRedTeamResults();
  const { loading: operationLoading, error: operationError, startTesting, stopTesting, runTest } = useRedTeamOperations();
  const { data: models, loading: modelsLoading, execute: executeModels } = useModels();

  // Load data when component mounts
  useEffect(() => {
    executeStatus();
    executeResults();
    executeModels();
  }, [executeStatus, executeResults, executeModels]);

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle test operations
  const handleStartTesting = async () => {
    try {
      setIsTestRunning(true);
      await startTesting();
      refetchStatus();
    } catch (error) {
      console.error('Failed to start testing:', error);
    } finally {
      setIsTestRunning(false);
    }
  };

  const handleStopTesting = async () => {
    try {
      await stopTesting();
      refetchStatus();
    } catch (error) {
      console.error('Failed to stop testing:', error);
    }
  };

  const handleRunTest = async () => {
    try {
      setIsTestRunning(true);
      await runTest(testConfig);
      refetchResults();
      setTestDialogOpen(false);
    } catch (error) {
      console.error('Failed to run test:', error);
    } finally {
      setIsTestRunning(false);
    }
  };

  // Get status color and icon
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'PASS': return 'success';
      case 'FAIL': return 'error';
      case 'CRITICAL': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'PASS': return <CheckCircle />;
      case 'FAIL': return <Error />;
      case 'CRITICAL': return <Dangerous />;
      default: return <Info />;
    }
  };

  // Mock data for charts
  const attackTrendData = [
    { name: '00:00', attacks: 12, detected: 10, detection_rate: 83.3 },
    { name: '04:00', attacks: 8, detected: 7, detection_rate: 87.5 },
    { name: '08:00', attacks: 15, detected: 12, detection_rate: 80.0 },
    { name: '12:00', attacks: 6, detected: 6, detection_rate: 100.0 },
    { name: '16:00', attacks: 10, detected: 9, detection_rate: 90.0 },
    { name: '20:00', attacks: 7, detected: 6, detection_rate: 85.7 },
  ];

  const attackCategoryData = [
    { name: 'Prompt Injection', value: 35, color: '#ff6b6b' },
    { name: 'Jailbreak', value: 25, color: '#4ecdc4' },
    { name: 'System Extraction', value: 20, color: '#45b7d1' },
    { name: 'Code Injection', value: 15, color: '#f9ca24' },
    { name: 'Other', value: 5, color: '#6c5ce7' },
  ];

  // Test result card component
  const TestResultCard: React.FC<{ result: RedTeamTestResult }> = ({ result }) => (
    <Card sx={{ height: '100%', position: 'relative' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Avatar sx={{ bgcolor: 'warning.main' }}>
              <Security />
            </Avatar>
            <Box>
              <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
                {result.model_name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {result.model_type} • {new Date(result.timestamp).toLocaleString()}
              </Typography>
            </Box>
          </Box>
          <Chip
            icon={getStatusIcon(result.overall_status)}
            label={result.overall_status}
            color={getStatusColor(result.overall_status) as any}
            size="small"
          />
        </Box>

        {/* Test Summary */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Test Summary
          </Typography>
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">
                Total Attacks: {result.total_attacks}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">
                Vulnerabilities: {result.vulnerabilities_found}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">
                Detection Rate: {result.detection_rate.toFixed(1)}%
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">
                Duration: {result.duration_ms}ms
              </Typography>
            </Grid>
          </Grid>
        </Box>

        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<Report />}
            onClick={() => setSelectedTest(result)}
          >
            Details
          </Button>
          
          {result.overall_status === 'FAIL' && (
            <Button
              size="small"
              variant="contained"
              startIcon={<School />}
              onClick={() => {
                console.log('Trigger retraining for:', result.model_name);
              }}
              color="warning"
            >
              Retrain
            </Button>
          )}
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold', mb: 1 }}>
            Red Team Testing
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Security testing and vulnerability assessment for ML models
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => {
              refetchStatus();
              refetchResults();
            }}
            disabled={statusLoading || resultsLoading}
          >
            Refresh
          </Button>
          {redTeamStatus?.running ? (
            <Button
              variant="contained"
              startIcon={<Stop />}
              onClick={handleStopTesting}
              disabled={operationLoading}
              color="error"
            >
              Stop Testing
            </Button>
          ) : (
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={() => setTestDialogOpen(true)}
              disabled={operationLoading}
            >
              Start Test
            </Button>
          )}
        </Box>
      </Box>

      {/* Status Alert */}
      {redTeamStatus?.running && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Red team testing is currently running. {redTeamStatus.total_tests} tests completed.
        </Alert>
      )}

      {/* Error Alert */}
      {operationError && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {operationError.error}
        </Alert>
      )}

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Test Results" />
          <Tab label="Security Dashboard" />
          <Tab label="Attack Analytics" />
        </Tabs>
      </Box>

      {/* Tab Panels */}
      <TabPanel value={tabValue} index={0}>
        {/* Test Results Grid */}
        {resultsLoading ? (
          <Grid container spacing={3}>
            {[...Array(6)].map((_, i) => (
              <Grid item xs={12} sm={6} md={4} key={i}>
                <Card sx={{ height: 400 }}>
                  <CardContent>
                    <Skeleton variant="rectangular" height={200} />
                    <Skeleton variant="text" height={40} />
                    <Skeleton variant="text" height={20} />
                    <Skeleton variant="text" height={20} />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        ) : (
          <Grid container spacing={3}>
            {redTeamResults?.results?.map((result, index) => (
              <Grid item xs={12} sm={6} md={4} key={result.test_id}>
                <Fade in timeout={300 + index * 100}>
                  <div>
                    <TestResultCard result={result} />
                  </div>
                </Fade>
              </Grid>
            ))}
          </Grid>
        )}

        {/* Empty State */}
        {!resultsLoading && (!redTeamResults?.results || redTeamResults.results.length === 0) && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Security sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No test results found
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              No security tests have been run yet
            </Typography>
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={() => setTestDialogOpen(true)}
            >
              Run Your First Security Test
            </Button>
          </Box>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {/* Security Dashboard */}
        <Typography variant="h6" gutterBottom>
          Security Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Real-time security monitoring and threat detection
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Attack Detection Trends
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={attackTrendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="attacks" stroke="#ff6b6b" name="Total Attacks" />
                    <Line type="monotone" dataKey="detected" stroke="#4ecdc4" name="Detected" />
                    <Line type="monotone" dataKey="detection_rate" stroke="#45b7d1" name="Detection Rate %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Attack Categories
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={attackCategoryData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      label
                    >
                      {attackCategoryData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        {/* Attack Analytics */}
        <Typography variant="h6" gutterBottom>
          Attack Analytics
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Detailed analysis of attack patterns and detection effectiveness
        </Typography>
        
        <Card>
          <CardContent>
            <Typography variant="body2" color="text.secondary">
              Attack analytics functionality will be available here
            </Typography>
          </CardContent>
        </Card>
      </TabPanel>

      {/* Test Configuration Dialog */}
      <Dialog open={testDialogOpen} onClose={() => setTestDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Configure Security Test</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel>Model to Test</InputLabel>
              <Select
                value={testConfig.model_name}
                label="Model to Test"
                onChange={(e) => setTestConfig(prev => ({ ...prev, model_name: e.target.value }))}
              >
                {models?.models && Object.values(models.models).filter((model: ModelInfo) => model.loaded).map((model: ModelInfo) => (
                  <MenuItem key={model.name} value={model.name}>
                    {model.name} ({model.type})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Number of Tests"
              type="number"
              value={testConfig.test_count}
              onChange={(e) => setTestConfig(prev => ({ ...prev, test_count: parseInt(e.target.value) }))}
              inputProps={{ min: 1, max: 100 }}
            />

            <FormControl fullWidth>
              <InputLabel>Attack Categories</InputLabel>
              <Select
                multiple
                value={testConfig.attack_categories}
                label="Attack Categories"
                onChange={(e) => setTestConfig(prev => ({ ...prev, attack_categories: e.target.value as string[] }))}
              >
                <MenuItem value="prompt_injection">Prompt Injection</MenuItem>
                <MenuItem value="jailbreak">Jailbreak</MenuItem>
                <MenuItem value="system_extraction">System Extraction</MenuItem>
                <MenuItem value="code_injection">Code Injection</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleRunTest}
            variant="contained"
            disabled={!testConfig.model_name || isTestRunning}
            startIcon={isTestRunning ? <CircularProgress size={20} /> : <PlayArrow />}
          >
            {isTestRunning ? 'Running Test...' : 'Run Test'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Test Details Dialog */}
      <Dialog open={!!selectedTest} onClose={() => setSelectedTest(null)} maxWidth="lg" fullWidth>
        <DialogTitle>Test Details - {selectedTest?.model_name}</DialogTitle>
        <DialogContent>
          {selectedTest && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Test Summary
              </Typography>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Total Attacks
                  </Typography>
                  <Typography variant="h6">{selectedTest.total_attacks}</Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Vulnerabilities Found
                  </Typography>
                  <Typography variant="h6" color="error.main">
                    {selectedTest.vulnerabilities_found}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Detection Rate
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {selectedTest.detection_rate.toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Overall Status
                  </Typography>
                  <Chip
                    label={selectedTest.overall_status}
                    color={getStatusColor(selectedTest.overall_status) as any}
                    icon={getStatusIcon(selectedTest.overall_status)}
                  />
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedTest(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default RedTeam;