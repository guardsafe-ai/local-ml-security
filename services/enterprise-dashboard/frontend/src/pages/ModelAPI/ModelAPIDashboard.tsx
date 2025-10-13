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
  Zoom,
  Snackbar,
  AccordionSummary as MuiAccordionSummary,
  AccordionDetails as MuiAccordionDetails
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Security,
  BugReport,
  CompareArrows,
  Timeline,
  Assessment,
  Refresh,
  Download,
  FilterList,
  ExpandMore,
  Warning,
  CheckCircle,
  Error,
  Info,
  Speed,
  Shield,
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
  Assessment as AssessmentIcon,
  Insights,
  Recommend,
  Flag,
  PrivacyTip,
  Gavel,
  PersonAdd,
  PersonRemove,
  DataUsage,
  Lock,
  LockOpen,
  VisibilityOff as VisibilityOffIcon,
  ContentCopy,
  Save,
  Upload,
  Download as DownloadIcon,
  Search,
  FilterList as FilterListIcon,
  Sort,
  MoreVert,
  Close,
  Check,
  Cancel,
  Warning as WarningIcon,
  Info as InfoIcon,
  Error as ErrorIcon,
  Success as SuccessIcon,
  SmartToy,
  Memory,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
  AutoAwesome,
  Compare,
  Timeline as TimelineIcon2,
  Assessment as AssessmentIcon2,
  Insights as InsightsIcon,
  Recommend as RecommendIcon,
  Flag as FlagIcon,
  PrivacyTip as PrivacyTipIcon,
  Gavel as GavelIcon,
  PersonAdd as PersonAddIcon,
  PersonRemove as PersonRemoveIcon,
  DataUsage as DataUsageIcon,
  Lock as LockIcon,
  LockOpen as LockOpenIcon,
  VisibilityOff as VisibilityOffIcon2,
  ContentCopy as ContentCopyIcon,
  Save as SaveIcon,
  Upload as UploadIcon,
  Download as DownloadIcon2,
  Search as SearchIcon,
  FilterList as FilterListIcon2,
  Sort as SortIcon,
  MoreVert as MoreVertIcon,
  Close as CloseIcon,
  Check as CheckIcon,
  Cancel as CancelIcon,
  Warning as WarningIcon2,
  Info as InfoIcon2,
  Error as ErrorIcon2,
  Success as SuccessIcon2
} from '@mui/icons-material';
import { useModelAPI } from '../../hooks/useModelAPI';
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
      id={`model-api-tabpanel-${index}`}
      aria-labelledby={`model-api-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const ModelAPIDashboard: React.FC = () => {
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
  const [predictionDialogOpen, setPredictionDialogOpen] = useState(false);
  const [predictionText, setPredictionText] = useState('');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [useEnsemble, setUseEnsemble] = useState(true);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  // API Hooks
  const {
    models,
    modelInfo,
    cacheStats,
    health,
    loading,
    error,
    refetch
  } = useModelAPI();

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
    console.log(`Exporting model API metrics in ${exportFormat} format`);
    setExportDialogOpen(false);
  };

  const handlePrediction = async () => {
    if (!predictionText.trim()) return;
    
    try {
      // This would call the actual API
      const result = {
        text: predictionText,
        prediction: "prompt_injection",
        confidence: 0.95,
        probabilities: {
          prompt_injection: 0.95,
          jailbreak: 0.02,
          system_extraction: 0.01,
          code_injection: 0.01,
          benign: 0.01
        },
        model_predictions: {
          distilbert_pretrained: {
            prediction: "prompt_injection",
            confidence: 0.95,
            probabilities: {
              prompt_injection: 0.95,
              jailbreak: 0.02,
              system_extraction: 0.01,
              code_injection: 0.01,
              benign: 0.01
            }
          }
        },
        ensemble_used: useEnsemble,
        processing_time_ms: 2.5,
        timestamp: new Date().toISOString()
      };
      setPredictionResult(result);
      setSnackbarMessage('Prediction completed successfully');
      setSnackbarOpen(true);
    } catch (error) {
      setSnackbarMessage('Error making prediction');
      setSnackbarOpen(true);
    }
  };

  const handleLoadModel = async (modelName: string) => {
    try {
      // This would call the actual API
      setSnackbarMessage(`Model ${modelName} loaded successfully`);
      setSnackbarOpen(true);
      refetch();
    } catch (error) {
      setSnackbarMessage(`Error loading model ${modelName}`);
      setSnackbarOpen(true);
    }
  };

  const handleUnloadModel = async (modelName: string) => {
    try {
      // This would call the actual API
      setSnackbarMessage(`Model ${modelName} unloaded successfully`);
      setSnackbarOpen(true);
      refetch();
    } catch (error) {
      setSnackbarMessage(`Error unloading model ${modelName}`);
      setSnackbarOpen(true);
    }
  };

  // Data processing functions
  const getModelStatusColor = (loaded: boolean) => {
    return loaded ? 'success' : 'default';
  };

  const getModelTypeColor = (type: string) => {
    switch (type) {
      case 'pytorch': return 'primary';
      case 'sklearn': return 'secondary';
      case 'mlflow': return 'info';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  // Render functions
  const renderOverviewCards = () => {
    if (loading) return <LoadingSkeleton variant="metrics" count={4} />;

    const totalModels = Object.keys(models || {}).length;
    const loadedModels = Object.values(models || {}).filter((model: any) => model.loaded).length;
    const avgConfidence = 0.87; // This would come from actual data
    const totalPredictions = 1247; // This would come from actual data

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Total Models
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {totalModels}
                  </Typography>
                </Box>
                <SmartToy sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Loaded Models
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {loadedModels}
                  </Typography>
                </Box>
                <Memory sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Avg Confidence
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {(avgConfidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <SpeedIcon sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Total Predictions
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {totalPredictions.toLocaleString()}
                  </Typography>
                </Box>
                <PsychologyIcon sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderModelManagement = () => {
    if (loading) return <LoadingSkeleton variant="table" count={5} />;

    const modelList = Object.values(models || {});

    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" component="h2">
              Model Management
            </Typography>
            <Box display="flex" gap={1}>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={handleRefresh}
              >
                Refresh All
              </Button>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setPredictionDialogOpen(true)}
              >
                Test Prediction
              </Button>
            </Box>
          </Box>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="center">Status</TableCell>
                  <TableCell>Source</TableCell>
                  <TableCell>Version</TableCell>
                  <TableCell align="right">Labels</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {modelList.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((model: any, index: number) => (
                  <TableRow key={index} hover>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: 'primary.main' }}>
                          <SmartToy />
                        </Avatar>
                        <Box>
                          <Typography variant="body2" fontWeight="medium">
                            {model.name}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            {model.path}
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={model.type} 
                        size="small" 
                        color={getModelTypeColor(model.type) as any}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={model.loaded ? 'Loaded' : 'Unloaded'} 
                        size="small" 
                        color={getModelStatusColor(model.loaded) as any}
                        variant="filled"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {model.model_source || 'Unknown'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {model.model_version || 'Unknown'}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Box display="flex" gap={0.5} flexWrap="wrap" justifyContent="flex-end">
                        {model.labels?.slice(0, 2).map((label: string, idx: number) => (
                          <Chip key={idx} label={label} size="small" variant="outlined" />
                        ))}
                        {model.labels?.length > 2 && (
                          <Chip label={`+${model.labels.length - 2}`} size="small" variant="outlined" />
                        )}
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Box display="flex" gap={0.5}>
                        <Tooltip title="View Details">
                          <IconButton size="small">
                            <Visibility />
                          </IconButton>
                        </Tooltip>
                        {model.loaded ? (
                          <Tooltip title="Unload Model">
                            <IconButton 
                              size="small" 
                              color="warning"
                              onClick={() => handleUnloadModel(model.name)}
                            >
                              <Pause />
                            </IconButton>
                          </Tooltip>
                        ) : (
                          <Tooltip title="Load Model">
                            <IconButton 
                              size="small" 
                              color="success"
                              onClick={() => handleLoadModel(model.name)}
                            >
                              <PlayArrow />
                            </IconButton>
                          </Tooltip>
                        )}
                        <Tooltip title="Reload Model">
                          <IconButton size="small" color="info">
                            <Refresh />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={modelList.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>
    );
  };

  const renderPredictionTool = () => {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            Model Prediction Tool
          </Typography>
          
          <Box mb={2}>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Text to Analyze"
              value={predictionText}
              onChange={(e) => setPredictionText(e.target.value)}
              placeholder="Enter text to analyze for security threats..."
              variant="outlined"
            />
          </Box>
          
          <Box mb={2}>
            <FormControl fullWidth>
              <InputLabel>Select Models</InputLabel>
              <Select
                multiple
                value={selectedModels}
                onChange={(e) => setSelectedModels(e.target.value as string[])}
                label="Select Models"
              >
                {Object.keys(models || {}).map((modelName) => (
                  <MenuItem key={modelName} value={modelName}>
                    {modelName}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          
          <Box display="flex" gap={2} mb={2}>
            <FormControlLabel
              control={
                <Switch 
                  checked={useEnsemble} 
                  onChange={(e) => setUseEnsemble(e.target.checked)}
                  color="primary"
                />
              }
              label="Use Ensemble"
            />
            <Button
              variant="contained"
              startIcon={<AutoAwesome />}
              onClick={handlePrediction}
              disabled={!predictionText.trim()}
            >
              Analyze Text
            </Button>
          </Box>
          
          {predictionResult && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Prediction Result:
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" fontWeight="medium">
                      Prediction: {predictionResult.prediction}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Confidence: {(predictionResult.confidence * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Processing Time: {predictionResult.processing_time_ms}ms
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" fontWeight="medium" gutterBottom>
                      Probabilities:
                    </Typography>
                    {Object.entries(predictionResult.probabilities || {}).map(([label, prob]) => (
                      <Box key={label} display="flex" justifyContent="space-between" mb={0.5}>
                        <Typography variant="body2">{label}:</Typography>
                        <Typography variant="body2" fontWeight="medium">
                          {((prob as number) * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    ))}
                  </Grid>
                </Grid>
              </Paper>
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderCacheStats = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const cacheData = [
      { metric: 'Memory Used', value: cacheStats?.memory_used || '0M' },
      { metric: 'Connected Clients', value: cacheStats?.connected_clients || 0 },
      { metric: 'Total Commands', value: cacheStats?.total_commands_processed || 0 },
      { metric: 'Cache Hits', value: cacheStats?.keyspace_hits || 0 },
      { metric: 'Cache Misses', value: cacheStats?.keyspace_misses || 0 }
    ];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Cache Performance
              </Typography>
              <BarChart 
                data={cacheData}
                title="Cache Metrics"
                dataKey="value"
                nameKey="metric"
                height={300}
                color="#4facfe"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Cache Status
              </Typography>
              
              <Box mb={3}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Redis Connection</Typography>
                  <Chip 
                    label={cacheStats?.redis_connected ? 'Connected' : 'Disconnected'} 
                    color={cacheStats?.redis_connected ? 'success' : 'error'}
                    variant="filled"
                  />
                </Box>
              </Box>

              <Box mb={3}>
                <Typography variant="body2" gutterBottom>
                  Memory Usage
                </Typography>
                <Typography variant="h6" color="primary.main">
                  {cacheStats?.memory_used || '0M'}
                </Typography>
              </Box>

              <Box mb={3}>
                <Typography variant="body2" gutterBottom>
                  Hit Rate
                </Typography>
                <Typography variant="h6" color="success.main">
                  {cacheStats?.keyspace_hits && cacheStats?.keyspace_misses ? 
                    `${((cacheStats.keyspace_hits / (cacheStats.keyspace_hits + cacheStats.keyspace_misses)) * 100).toFixed(1)}%` : 
                    '0%'
                  }
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderModelPerformance = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const performanceData = [
      { model: 'DistilBERT', accuracy: 0.95, speed: 2.5, memory: 512 },
      { model: 'BERT Base', accuracy: 0.97, speed: 4.2, memory: 1024 },
      { model: 'RoBERTa', accuracy: 0.96, speed: 3.8, memory: 768 },
      { model: 'DeBERTa v3', accuracy: 0.98, speed: 5.1, memory: 1536 }
    ];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Model Performance Comparison
              </Typography>
              <BarChart 
                data={performanceData}
                title="Model Accuracy"
                dataKey="accuracy"
                nameKey="model"
                height={400}
                color="#667eea"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Processing Speed
              </Typography>
              <BarChart 
                data={performanceData}
                title="Speed (ms)"
                dataKey="speed"
                nameKey="model"
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
                Memory Usage
              </Typography>
              <BarChart 
                data={performanceData}
                title="Memory (MB)"
                dataKey="memory"
                nameKey="model"
                height={300}
                color="#43e97b"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  return (
    <ErrorBoundary>
      <Box sx={{ width: '100%' }}>
        {/* Header Controls */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4" component="h1" gutterBottom>
            Model API Dashboard
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
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="model api tabs">
            <Tab label="Model Management" icon={<SmartToy />} />
            <Tab label="Prediction Tool" icon={<AutoAwesome />} />
            <Tab label="Cache Stats" icon={<Memory />} />
            <Tab label="Performance" icon={<SpeedIcon />} />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            {renderModelManagement()}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {renderPredictionTool()}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {renderCacheStats()}
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {renderModelPerformance()}
          </TabPanel>
        </Box>

        {/* Export Dialog */}
        <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)}>
          <DialogTitle>Export Model API Metrics</DialogTitle>
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

        {/* Snackbar */}
        <Snackbar
          open={snackbarOpen}
          autoHideDuration={6000}
          onClose={() => setSnackbarOpen(false)}
          message={snackbarMessage}
        />
      </Box>
    </ErrorBoundary>
  );
};

export default ModelAPIDashboard;
