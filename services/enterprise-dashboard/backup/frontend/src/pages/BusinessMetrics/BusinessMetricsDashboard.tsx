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
  Alert,
  Button,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Badge,
  Avatar,
  Stack,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  CircularProgress,
  Fade,
  Zoom
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AttachMoney,
  Speed,
  Warning,
  CheckCircle,
  Error,
  Info,
  Refresh,
  Download,
  FilterList,
  ExpandMore,
  Analytics as AnalyticsIcon,
  ModelTraining,
  Psychology,
  AutoGraph,
  Storage,
  Timeline as TimelineIcon,
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
  TableChart,
  ViewList,
  Settings,
  PlayArrow,
  Pause,
  Stop,
  Add,
  Edit,
  Delete,
  Visibility,
  VisibilityOff,
  BusinessCenter,
  AccountBalance,
  TrendingFlat,
  ShowChart,
  Assessment,
  Insights,
  Recommend,
  Flag,
  Security,
  BugReport,
  CompareArrows,
  Timeline,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { useBusinessMetrics } from '../../hooks/useBusinessMetrics';
import { PerformanceChart, BarChart, PieChart } from '../../components/Charts';
import { LoadingSkeleton } from '../../components/LoadingSkeleton';
import { ErrorBoundary } from '../../components/ErrorBoundary';

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
      id={`business-metrics-tabpanel-${index}`}
      aria-labelledby={`business-metrics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const BusinessMetricsDashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState(7);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [expandedPanel, setExpandedPanel] = useState<string | false>(false);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState('json');
  const [activeStep, setActiveStep] = useState(0);

  // API Hooks
  const {
    kpis,
    attackSuccessRate,
    modelDrift,
    costMetrics,
    systemEffectiveness,
    recommendations,
    loading,
    error,
    refetch,
    execute
  } = useBusinessMetrics(timeRange);

  // Load data when component mounts
  useEffect(() => {
    execute();
  }, [execute]);

  // Auto-refresh effect - DISABLED to prevent excessive polling
  // useEffect(() => {
  //   if (!autoRefresh) return;
  //   
  //   const interval = setInterval(() => {
  //     refetch();
  //   }, 30000); // 30 seconds

  //   return () => clearInterval(interval);
  // }, [autoRefresh, refetch]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleTimeRangeChange = (event: any) => {
    setTimeRange(event.target.value);
  };

  const handleRefresh = () => {
    refetch();
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handlePanelChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedPanel(isExpanded ? panel : false);
  };

  const handleExport = () => {
    setExportDialogOpen(true);
  };

  const handleExportConfirm = () => {
    console.log(`Exporting business metrics in ${exportFormat} format`);
    setExportDialogOpen(false);
  };

  // Data processing functions
  const getEffectivenessColor = (rate: number) => {
    if (rate >= 0.8) return 'success';
    if (rate >= 0.6) return 'warning';
    return 'error';
  };

  const getDriftSeverity = (severity: string) => {
    switch (severity) {
      case 'critical': return { color: 'error', icon: <Error /> };
      case 'high': return { color: 'warning', icon: <Warning /> };
      case 'medium': return { color: 'info', icon: <Info /> };
      case 'low': return { color: 'success', icon: <CheckCircle /> };
      default: return { color: 'default', icon: <Info /> };
    }
  };

  const getTrendIcon = (trend: number) => {
    if (trend > 0) return <TrendingUp color="success" />;
    if (trend < 0) return <TrendingDown color="error" />;
    return <TrendingFlat color="info" />;
  };

  // Render functions
  const renderOverviewCards = () => {
    if (loading) return <LoadingSkeleton variant="metrics" count={4} />;

    const totalCost = costMetrics?.total_cost_usd || 0;
    const effectiveness = systemEffectiveness?.overall_effectiveness || 0;
    const attackRate = attackSuccessRate?.success_rate || 0;
    const driftCount = modelDrift?.length || 0;

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Total Cost
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    ${totalCost.toFixed(2)}
                  </Typography>
                </Box>
                <AttachMoney sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    System Effectiveness
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {(effectiveness * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Speed sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Attack Success Rate
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {(attackRate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Security sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Models with Drift
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {driftCount}
                  </Typography>
                </Box>
                <Warning sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderCostAnalysis = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const costData = [
      { name: 'Compute', value: costMetrics?.compute_cost || 0, color: '#667eea' },
      { name: 'Storage', value: costMetrics?.storage_cost || 0, color: '#f093fb' },
      { name: 'API Calls', value: costMetrics?.api_calls_cost || 0, color: '#4facfe' },
      { name: 'Training', value: costMetrics?.model_training_cost || 0, color: '#43e97b' }
    ];

    const trendData = [
      { period: '7d', trend: costMetrics?.cost_trend_7d || 0 },
      { period: '30d', trend: costMetrics?.cost_trend_30d || 0 }
    ];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Cost Breakdown
              </Typography>
              <PieChart 
                data={costData}
                title="Cost Distribution"
                dataKey="value"
                nameKey="name"
                height={300}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Cost Trends
              </Typography>
              <BarChart 
                data={trendData}
                title="Cost Trends"
                dataKey="trend"
                nameKey="period"
                height={300}
                color="#4facfe"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Cost Metrics Summary
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.50' }}>
                    <Typography variant="h4" color="primary.main" gutterBottom>
                      ${costMetrics?.total_cost_usd?.toFixed(2) || '0.00'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Total Cost
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.50' }}>
                    <Typography variant="h4" color="success.main" gutterBottom>
                      ${costMetrics?.cost_per_prediction?.toFixed(4) || '0.0000'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Cost per Prediction
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.50' }}>
                    <Typography variant="h4" color="info.main" gutterBottom>
                      {costMetrics?.cost_trend_7d ? 
                        `${costMetrics.cost_trend_7d > 0 ? '+' : ''}${(costMetrics.cost_trend_7d * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      7-Day Trend
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.50' }}>
                    <Typography variant="h4" color="warning.main" gutterBottom>
                      {costMetrics?.cost_trend_30d ? 
                        `${costMetrics.cost_trend_30d > 0 ? '+' : ''}${(costMetrics.cost_trend_30d * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      30-Day Trend
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderSystemEffectiveness = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const effectivenessData = [
      { metric: 'Detection Accuracy', value: systemEffectiveness?.detection_accuracy || 0 },
      { metric: 'False Positive Rate', value: systemEffectiveness?.false_positive_rate || 0 },
      { metric: 'False Negative Rate', value: systemEffectiveness?.false_negative_rate || 0 },
      { metric: 'User Satisfaction', value: systemEffectiveness?.user_satisfaction_score || 0 }
    ];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                System Effectiveness Metrics
              </Typography>
              <BarChart 
                data={effectivenessData}
                title="Effectiveness Metrics"
                dataKey="value"
                nameKey="metric"
                height={400}
                color="#667eea"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Key Performance Indicators
              </Typography>
              
              <Box mb={3}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Overall Effectiveness</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {((systemEffectiveness?.overall_effectiveness || 0) * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={(systemEffectiveness?.overall_effectiveness || 0) * 100}
                  color={getEffectivenessColor(systemEffectiveness?.overall_effectiveness || 0)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box mb={3}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Response Time (P95)</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {systemEffectiveness?.response_time_p95?.toFixed(2) || '0.00'}ms
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min((systemEffectiveness?.response_time_p95 || 0) / 1000 * 100, 100)}
                  color="info"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box mb={3}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Availability</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(systemEffectiveness?.availability_percent || 0).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={systemEffectiveness?.availability_percent || 0}
                  color="success"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderModelDrift = () => {
    if (loading) return <LoadingSkeleton variant="table" count={5} />;

    if (!modelDrift || modelDrift.length === 0) {
      return (
        <Alert severity="success">
          No model drift detected. All models are performing within expected parameters.
        </Alert>
      );
    }

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            Model Drift Detection
          </Typography>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell align="center">Drift Detected</TableCell>
                  <TableCell align="right">Drift Score</TableCell>
                  <TableCell align="center">Severity</TableCell>
                  <TableCell align="right">Confidence</TableCell>
                  <TableCell align="right">Last Check</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {modelDrift.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((drift: any, index: number) => {
                  const severity = getDriftSeverity(drift.severity);
                  
                  return (
                    <TableRow key={index} hover>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: 'primary.main' }}>
                            <ModelTraining />
                          </Avatar>
                          <Typography variant="body2" fontWeight="medium">
                            {drift.model_name}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Chip 
                          label={drift.drift_detected ? 'Yes' : 'No'} 
                          size="small" 
                          color={drift.drift_detected ? 'error' : 'success'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" fontWeight="medium">
                          {drift.drift_score.toFixed(3)}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Box display="flex" alignItems="center" justifyContent="center">
                          <Chip 
                            label={drift.severity} 
                            size="small" 
                            color={severity.color as any}
                            variant="filled"
                            icon={severity.icon}
                          />
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="textSecondary">
                          {drift.confidence_interval ? 
                            `${(drift.confidence_interval[0] * 100).toFixed(1)}-${(drift.confidence_interval[1] * 100).toFixed(1)}%` : 
                            'N/A'
                          }
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="textSecondary">
                          {new Date(drift.last_drift_check).toLocaleDateString()}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Box display="flex" gap={0.5}>
                          <Tooltip title="View Details">
                            <IconButton size="small">
                              <Visibility />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Retrain Model">
                            <IconButton size="small" color="warning">
                              <ModelTraining />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
          
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={modelDrift.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>
    );
  };

  const renderAttackSuccessRate = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const attackData = [
      { category: 'Prompt Injection', rate: attackSuccessRate?.by_category?.['prompt_injection'] || 0 },
      { category: 'Jailbreak', rate: attackSuccessRate?.by_category?.['jailbreak'] || 0 },
      { category: 'System Extraction', rate: attackSuccessRate?.by_category?.['system_extraction'] || 0 },
      { category: 'Code Injection', rate: attackSuccessRate?.by_category?.['code_injection'] || 0 }
    ];

    const modelData = Object.entries(attackSuccessRate?.by_model || {}).map(([model, rate]) => ({
      model,
      rate: rate as number
    }));

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Attack Success by Category
              </Typography>
              <BarChart 
                data={attackData}
                title="Success Rate by Attack Category"
                dataKey="rate"
                nameKey="category"
                height={300}
                color="#f093fb"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Attack Success by Model
              </Typography>
              <BarChart 
                data={modelData}
                title="Success Rate by Model"
                dataKey="rate"
                nameKey="model"
                height={300}
                color="#4facfe"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Attack Success Rate Summary
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.50' }}>
                    <Typography variant="h4" color="primary.main" gutterBottom>
                      {attackSuccessRate?.total_attacks || 0}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Total Attacks
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.50' }}>
                    <Typography variant="h4" color="success.main" gutterBottom>
                      {attackSuccessRate?.successful_attacks || 0}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Successful Attacks
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.50' }}>
                    <Typography variant="h4" color="info.main" gutterBottom>
                      {((attackSuccessRate?.success_rate || 0) * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Success Rate
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.50' }}>
                    <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                      {getTrendIcon(attackSuccessRate?.trend_7d || 0)}
                    </Box>
                    <Typography variant="h6" color="warning.main" gutterBottom>
                      {attackSuccessRate?.trend_7d ? 
                        `${attackSuccessRate.trend_7d > 0 ? '+' : ''}${(attackSuccessRate.trend_7d * 100).toFixed(1)}%` : 
                        '0.0%'
                      }
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      7-Day Trend
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderRecommendations = () => {
    if (loading) return <LoadingSkeleton variant="list" count={5} />;

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            AI-Powered Recommendations
          </Typography>
          
          <List>
            {recommendations?.map((recommendation: string, index: number) => (
              <ListItem key={index} divider>
                <ListItemIcon>
                  <Recommend color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary={recommendation}
                  secondary={`Priority: ${index < 2 ? 'High' : index < 4 ? 'Medium' : 'Low'}`}
                />
                <Button size="small" variant="outlined">
                  Implement
                </Button>
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>
    );
  };

  return (
    <ErrorBoundary>
      <Box sx={{ width: '100%' }}>
        {/* Header Controls */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4" component="h1" gutterBottom>
            Business Metrics Dashboard
          </Typography>
          
          <Box display="flex" alignItems="center" gap={2}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select value={timeRange} onChange={handleTimeRangeChange} label="Time Range">
                <MenuItem value={1}>Last 24h</MenuItem>
                <MenuItem value={7}>Last 7 days</MenuItem>
                <MenuItem value={30}>Last 30 days</MenuItem>
                <MenuItem value={90}>Last 90 days</MenuItem>
              </Select>
            </FormControl>
            
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
              <IconButton onClick={handleRefresh} color="primary">
                <Refresh />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Export Data">
              <IconButton onClick={handleExport} color="primary">
                <Download />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Overview Cards */}
        {renderOverviewCards()}

        <Box sx={{ mt: 3 }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="business metrics tabs">
            <Tab label="Cost Analysis" icon={<AttachMoney />} />
            <Tab label="System Effectiveness" icon={<Speed />} />
            <Tab label="Model Drift" icon={<Warning />} />
            <Tab label="Attack Success" icon={<Security />} />
            <Tab label="Recommendations" icon={<Recommend />} />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            {renderCostAnalysis()}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {renderSystemEffectiveness()}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {renderModelDrift()}
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {renderAttackSuccessRate()}
          </TabPanel>

          <TabPanel value={tabValue} index={4}>
            {renderRecommendations()}
          </TabPanel>
        </Box>

        {/* Export Dialog */}
        <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)}>
          <DialogTitle>Export Business Metrics</DialogTitle>
          <DialogContent>
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel>Export Format</InputLabel>
              <Select
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value)}
                label="Export Format"
              >
                <MenuItem value="json">JSON</MenuItem>
                <MenuItem value="csv">CSV</MenuItem>
                <MenuItem value="xlsx">Excel</MenuItem>
                <MenuItem value="pdf">PDF Report</MenuItem>
              </Select>
            </FormControl>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setExportDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleExportConfirm} variant="contained">
              Export
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </ErrorBoundary>
  );
};

export default BusinessMetricsDashboard;
