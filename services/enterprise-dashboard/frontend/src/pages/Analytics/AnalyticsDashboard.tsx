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
  StepContent
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
  VisibilityOff
} from '@mui/icons-material';
import { useAnalyticsDashboard, useStoreRedTeamResults, useStoreModelPerformance } from '../../hooks/useAnalytics';
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
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const AnalyticsDashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState(7);
  const [selectedModel, setSelectedModelLocal] = useState('all');
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
    redTeamData,
    trendsData,
    comparisonData,
    selectedModel: currentSelectedModel,
    setSelectedModel,
    loading,
    error,
    refetch
  } = useAnalyticsDashboard(timeRange);

  const { storeResults, loading: storeLoading } = useStoreRedTeamResults();
  const { storePerformance, loading: performanceLoading } = useStoreModelPerformance();

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

  const handleModelChange = (event: any) => {
    setSelectedModelLocal(event.target.value);
    setSelectedModelLocal(event.target.value);
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
    // Export logic here
    console.log(`Exporting data in ${exportFormat} format`);
    setExportDialogOpen(false);
  };

  // Data processing functions
  const getDetectionRateColor = (rate: number) => {
    if (rate >= 0.8) return 'success';
    if (rate >= 0.6) return 'warning';
    return 'error';
  };

  const getVulnerabilitySeverity = (count: number) => {
    if (count === 0) return { label: 'None', color: 'success' };
    if (count <= 2) return { label: 'Low', color: 'info' };
    if (count <= 5) return { label: 'Medium', color: 'warning' };
    return { label: 'High', color: 'error' };
  };

  const calculateImprovement = (pretrained: number, trained: number) => {
    if (!pretrained || !trained) return 0;
    return ((trained - pretrained) / pretrained) * 100;
  };

  // Render functions
  const renderOverviewCards = () => {
    if (loading) return <LoadingSkeleton variant="metrics" count={4} />;

    const totalTests = redTeamData?.summary?.reduce((sum: number, item: any) => sum + item.total_tests, 0) || 0;
    const avgDetectionRate = redTeamData?.summary?.reduce((sum: number, item: any) => sum + parseFloat(item.avg_detection_rate || 0), 0) / (redTeamData?.summary?.length || 1) || 0;
    const totalVulnerabilities = redTeamData?.summary?.reduce((sum: number, item: any) => sum + parseFloat(item.avg_vulnerabilities || 0), 0) || 0;
    const modelsTested = redTeamData?.summary?.length || 0;

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Total Tests
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {totalTests.toLocaleString()}
                  </Typography>
                </Box>
                <BugReport sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Detection Rate
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {(avgDetectionRate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Security sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Vulnerabilities
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {totalVulnerabilities.toFixed(0)}
                  </Typography>
                </Box>
                <Warning sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Models Tested
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {modelsTested}
                  </Typography>
                </Box>
                <ModelTraining sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderRedTeamSummary = () => {
    if (loading) return <LoadingSkeleton variant="table" count={5} />;

    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" component="h2">
              Red Team Test Summary
            </Typography>
            <Box display="flex" gap={1}>
              <Chip 
                label={`Last ${timeRange} days`} 
                color="primary" 
                variant="outlined"
              />
              <Tooltip title="Advanced Filters">
                <IconButton 
                  size="small" 
                  onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                >
                  <FilterList />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
          
          {showAdvancedFilters && (
            <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
              <Typography variant="subtitle2" gutterBottom>
                Advanced Filters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Detection Rate</InputLabel>
                    <Select value="" label="Detection Rate">
                      <MenuItem value="high">High (≥80%)</MenuItem>
                      <MenuItem value="medium">Medium (60-80%)</MenuItem>
                      <MenuItem value="low">Low (&lt;60%)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Vulnerability Level</InputLabel>
                    <Select value="" label="Vulnerability Level">
                      <MenuItem value="none">None</MenuItem>
                      <MenuItem value="low">Low (1-2)</MenuItem>
                      <MenuItem value="medium">Medium (3-5)</MenuItem>
                      <MenuItem value="high">High (&gt;5)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Model Type</InputLabel>
                    <Select value="" label="Model Type">
                      <MenuItem value="all">All Types</MenuItem>
                      <MenuItem value="pre-trained">Pre-trained</MenuItem>
                      <MenuItem value="trained">Trained</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Paper>
          )}
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="right">Tests</TableCell>
                  <TableCell align="right">Detection Rate</TableCell>
                  <TableCell align="right">Vulnerabilities</TableCell>
                  <TableCell align="right">Last Test</TableCell>
                  <TableCell align="center">Status</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {redTeamData?.summary?.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((item: any, index: number) => {
                  const detectionRate = parseFloat(item.avg_detection_rate || 0);
                  const vulnerabilityCount = parseFloat(item.avg_vulnerabilities || 0);
                  const severity = getVulnerabilitySeverity(vulnerabilityCount);
                  
                  return (
                    <TableRow key={index} hover>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: 'primary.main' }}>
                            <Psychology />
                          </Avatar>
                          <Typography variant="body2" fontWeight="medium">
                            {item.model_name}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={item.model_type} 
                          size="small" 
                          color={item.model_type === 'trained' ? 'success' : 'default'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" fontWeight="medium">
                          {item.total_tests}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Box display="flex" alignItems="center" justifyContent="flex-end">
                          <Typography 
                            variant="body2" 
                            color={`${getDetectionRateColor(detectionRate)}.main`}
                            fontWeight="medium"
                            mr={1}
                          >
                            {(detectionRate * 100).toFixed(1)}%
                          </Typography>
                          {detectionRate >= 0.8 ? (
                            <TrendingUp color="success" fontSize="small" />
                          ) : (
                            <TrendingDown color="error" fontSize="small" />
                          )}
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Chip 
                          label={vulnerabilityCount.toFixed(1)} 
                          size="small" 
                          color={severity.color as any}
                          variant="filled"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="textSecondary">
                          {new Date(item.last_test).toLocaleDateString()}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Tooltip title={severity.label}>
                          <Badge 
                            color={severity.color as any} 
                            variant="dot"
                            sx={{ '& .MuiBadge-badge': { width: 12, height: 12 } }}
                          >
                            <CheckCircle 
                              color={detectionRate >= 0.8 ? 'success' : 'warning'} 
                              fontSize="small" 
                            />
                          </Badge>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="center">
                        <Box display="flex" gap={0.5}>
                          <Tooltip title="View Details">
                            <IconButton size="small">
                              <Visibility />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Compare Models">
                            <IconButton size="small">
                              <CompareArrows />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Export Data">
                            <IconButton size="small">
                              <Download />
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
            count={redTeamData?.summary?.length || 0}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>
    );
  };

  const renderPerformanceTrends = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const chartData = trendsData?.trends?.map((item: any) => ({
      date: item.test_date,
      detectionRate: parseFloat(item.avg_detection_rate || 0) * 100,
      testCount: item.test_count,
      model: item.model_name,
      type: item.model_type
    })) || [];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Performance Trends Over Time
              </Typography>
              <PerformanceChart 
                data={chartData}
                title="Detection Rate Trends"
                dataKey="detectionRate"
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
                Test Volume Trends
              </Typography>
              <BarChart 
                data={chartData}
                title="Test Count by Date"
                dataKey="testCount"
                nameKey="date"
                height={300}
                color="#4facfe"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Model Distribution
              </Typography>
              <PieChart 
                data={chartData.reduce((acc: any[], item: any) => {
                  const existing = acc.find(x => x.model === item.model);
                  if (existing) {
                    existing.tests += item.testCount;
                  } else {
                    acc.push({ model: item.model, tests: item.testCount });
                  }
                  return acc;
                }, [])}
                title="Tests by Model"
                dataKey="tests"
                nameKey="model"
                height={300}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderModelComparison = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const { pretrained, trained, improvement } = comparisonData || {};

    if (!pretrained && !trained) {
      return (
        <Alert severity="info">
          No comparison data available for the selected model and time range.
        </Alert>
      );
    }

    const comparisonData_chart = [
      {
        name: 'Pre-trained',
        detectionRate: parseFloat(pretrained?.avg_detection_rate || 0) * 100,
        vulnerabilities: parseFloat(pretrained?.avg_vulnerabilities || 0),
        tests: pretrained?.test_count || 0
      },
      {
        name: 'Trained',
        detectionRate: parseFloat(trained?.avg_detection_rate || 0) * 100,
        vulnerabilities: parseFloat(trained?.avg_vulnerabilities || 0),
        tests: trained?.test_count || 0
      }
    ];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Detection Rate Comparison
              </Typography>
              <BarChart 
                data={comparisonData_chart}
                title="Detection Rate"
                dataKey="detectionRate"
                nameKey="name"
                height={300}
                color="#4facfe"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Vulnerability Detection
              </Typography>
              <BarChart 
                data={comparisonData_chart}
                title="Vulnerabilities Found"
                dataKey="vulnerabilities"
                nameKey="name"
                height={300}
                color="#f093fb"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Performance Improvement Analysis
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.50' }}>
                    <Typography variant="h4" color="primary.main" gutterBottom>
                      {improvement?.detection_rate_improvement ? 
                        `+${(improvement.detection_rate_improvement * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Detection Rate Improvement
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.50' }}>
                    <Typography variant="h4" color="success.main" gutterBottom>
                      {improvement?.vulnerability_detection_improvement ? 
                        `+${improvement.vulnerability_detection_improvement.toFixed(1)}` : 
                        'N/A'
                      }
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Vulnerability Detection Improvement
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.50' }}>
                    <Typography variant="h4" color="info.main" gutterBottom>
                      {trained?.test_count || 0}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Trained Model Tests
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

  const renderSecurityInsights = () => {
    const insights = [
      {
        title: "High-Risk Models",
        description: "Models with detection rates below 60%",
        count: redTeamData?.summary?.filter((item: any) => parseFloat(item.avg_detection_rate || 0) < 0.6).length || 0,
        severity: "error",
        icon: <Warning />,
        action: "Retrain immediately"
      },
      {
        title: "Vulnerability Hotspots",
        description: "Models with high vulnerability counts",
        count: redTeamData?.summary?.filter((item: any) => parseFloat(item.avg_vulnerabilities || 0) > 5).length || 0,
        severity: "warning",
        icon: <BugReport />,
        action: "Investigate patterns"
      },
      {
        title: "Well-Performing Models",
        description: "Models with detection rates above 80%",
        count: redTeamData?.summary?.filter((item: any) => parseFloat(item.avg_detection_rate || 0) > 0.8).length || 0,
        severity: "success",
        icon: <CheckCircle />,
        action: "Use as baseline"
      },
      {
        title: "Training Opportunities",
        description: "Models that could benefit from retraining",
        count: redTeamData?.summary?.filter((item: any) => 
          item.model_type === 'pre-trained' && parseFloat(item.avg_detection_rate || 0) < 0.7
        ).length || 0,
        severity: "info",
        icon: <ModelTraining />,
        action: "Schedule retraining"
      }
    ];

    return (
      <Grid container spacing={3}>
        {insights.map((insight, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card 
              sx={{ 
                height: '100%',
                border: `2px solid`,
                borderColor: `${insight.severity}.main`,
                bgcolor: `${insight.severity}.50`
              }}
            >
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Box color={`${insight.severity}.main`}>
                    {insight.icon}
                  </Box>
                  <Chip 
                    label={insight.action} 
                    size="small" 
                    color={insight.severity as any}
                    variant="outlined"
                  />
                </Box>
                
                <Typography variant="h4" color={`${insight.severity}.main`} gutterBottom>
                  {insight.count}
                </Typography>
                
                <Typography variant="body2" fontWeight="medium" gutterBottom>
                  {insight.title}
                </Typography>
                
                <Typography variant="caption" color="textSecondary" display="block" mb={2}>
                  {insight.description}
                </Typography>
                
                <Button 
                  size="small" 
                  color={insight.severity as any}
                  variant="outlined"
                  fullWidth
                >
                  Take Action
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
        
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Recommended Actions
              </Typography>
              
              <Stepper activeStep={activeStep} orientation="vertical">
                <Step>
                  <StepLabel>Review High-Risk Models</StepLabel>
                  <StepContent>
                    <Typography>
                      Identify models with detection rates below 60% and prioritize them for immediate retraining.
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Button
                        variant="contained"
                        onClick={() => setActiveStep(1)}
                        sx={{ mt: 1, mr: 1 }}
                      >
                        Continue
                      </Button>
                    </Box>
                  </StepContent>
                </Step>
                
                <Step>
                  <StepLabel>Analyze Vulnerability Patterns</StepLabel>
                  <StepContent>
                    <Typography>
                      Investigate common attack patterns that are bypassing current models.
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Button
                        variant="contained"
                        onClick={() => setActiveStep(2)}
                        sx={{ mt: 1, mr: 1 }}
                      >
                        Continue
                      </Button>
                    </Box>
                  </StepContent>
                </Step>
                
                <Step>
                  <StepLabel>Schedule Retraining</StepLabel>
                  <StepContent>
                    <Typography>
                      Set up automated retraining pipeline for models that need improvement.
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Button
                        variant="contained"
                        onClick={() => setActiveStep(3)}
                        sx={{ mt: 1, mr: 1 }}
                      >
                        Complete
                      </Button>
                    </Box>
                  </StepContent>
                </Step>
              </Stepper>
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
            Analytics Dashboard
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
            
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Model</InputLabel>
              <Select value={selectedModel} onChange={handleModelChange} label="Model">
                <MenuItem value="all">All Models</MenuItem>
                <MenuItem value="distilbert">DistilBERT</MenuItem>
                <MenuItem value="bert-base">BERT Base</MenuItem>
                <MenuItem value="roberta-base">RoBERTa Base</MenuItem>
                <MenuItem value="deberta-v3-base">DeBERTa v3</MenuItem>
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
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="analytics tabs">
            <Tab label="Summary" icon={<Assessment />} />
            <Tab label="Trends" icon={<Timeline />} />
            <Tab label="Comparison" icon={<CompareArrows />} />
            <Tab label="Insights" icon={<AnalyticsIcon />} />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            {renderRedTeamSummary()}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {renderPerformanceTrends()}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {renderModelComparison()}
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {renderSecurityInsights()}
          </TabPanel>
        </Box>

        {/* Export Dialog */}
        <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)}>
          <DialogTitle>Export Analytics Data</DialogTitle>
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

export default AnalyticsDashboard;
