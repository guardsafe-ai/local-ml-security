import React, { useState } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  IconButton,
  TextField,
  InputAdornment,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Badge,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Slider,
  FormGroup,
  Checkbox,
} from '@mui/material';
import {
  Analytics,
  TrendingUp,
  TrendingDown,
  AutoFixHigh,
  Assessment,
  Timeline,
  Speed,
  Memory,
  AttachMoney,
  Report,
  Search,
  FilterList,
  Download,
  Refresh,
  Add,
  Settings,
  Visibility,
  Edit,
  Delete,
  CheckCircle,
  Error,
  Warning,
  Info,
  ExpandMore,
  PlayArrow,
  Stop,
  Pause,
  Schedule,
  Notifications,
  Insights,
  ShowChart,
  BarChart,
  PieChart,
  LineChart,
} from '@mui/icons-material';
import MetricCard from '../../components/MetricCard/MetricCard';
import { LineChart as RechartsLineChart, BarChart as RechartsBarChart, PieChart as RechartsPieChart } from '../../components/Charts';

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
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Mock data
const mockPerformanceMetrics = [
  {
    id: '1',
    modelName: 'BERT Security',
    accuracy: 94.2,
    precision: 92.8,
    recall: 89.5,
    f1Score: 91.1,
    latency: 45,
    throughput: 1200,
    driftScore: 0.12,
    lastUpdated: '2024-01-20 14:30:00',
    status: 'healthy',
  },
  {
    id: '2',
    modelName: 'DistilBERT',
    accuracy: 91.5,
    precision: 89.2,
    recall: 87.8,
    f1Score: 88.5,
    latency: 28,
    throughput: 2100,
    driftScore: 0.08,
    lastUpdated: '2024-01-20 13:45:00',
    status: 'healthy',
  },
  {
    id: '3',
    modelName: 'RoBERTa Large',
    accuracy: 96.8,
    precision: 95.1,
    recall: 93.2,
    f1Score: 94.1,
    latency: 78,
    throughput: 850,
    driftScore: 0.25,
    lastUpdated: '2024-01-20 12:15:00',
    status: 'warning',
  },
];

const mockDriftAlerts = [
  {
    id: '1',
    modelName: 'RoBERTa Large',
    driftType: 'Data Drift',
    severity: 'high',
    driftScore: 0.25,
    threshold: 0.15,
    detectedAt: '2024-01-20 12:15:00',
    status: 'active',
    description: 'Significant drift detected in input data distribution',
  },
  {
    id: '2',
    modelName: 'BERT Security',
    driftType: 'Concept Drift',
    severity: 'medium',
    driftScore: 0.12,
    threshold: 0.10,
    detectedAt: '2024-01-19 16:30:00',
    status: 'acknowledged',
    description: 'Model performance degradation detected',
  },
];

const mockCostMetrics = [
  {
    service: 'Training',
    cost: 245.50,
    usage: '12 hours',
    trend: 15.2,
    period: 'This Month',
  },
  {
    service: 'Inference',
    cost: 189.30,
    usage: '45,000 requests',
    trend: -8.5,
    period: 'This Month',
  },
  {
    service: 'Storage',
    cost: 67.80,
    usage: '2.4 TB',
    trend: 3.2,
    period: 'This Month',
  },
  {
    service: 'Compute',
    cost: 156.20,
    usage: '8.5 hours',
    trend: 22.1,
    period: 'This Month',
  },
];

const mockAutoRetrainJobs = [
  {
    id: '1',
    modelName: 'BERT Security',
    trigger: 'Drift Detection',
    status: 'scheduled',
    scheduledAt: '2024-01-21 02:00:00',
    estimatedDuration: '4 hours',
    priority: 'high',
  },
  {
    id: '2',
    modelName: 'DistilBERT',
    trigger: 'Performance Drop',
    status: 'running',
    startedAt: '2024-01-20 18:30:00',
    estimatedDuration: '2 hours',
    priority: 'medium',
  },
  {
    id: '3',
    modelName: 'RoBERTa Large',
    trigger: 'Scheduled',
    status: 'completed',
    completedAt: '2024-01-20 14:45:00',
    duration: '3.5 hours',
    priority: 'low',
  },
];

const Analytics: React.FC = () => {
  const [value, setValue] = useState(0);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [autoRetrainDialogOpen, setAutoRetrainDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [reportType, setReportType] = useState('');
  const [dateRange, setDateRange] = useState('7d');

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const handleGenerateReport = () => {
    console.log('Generating report:', { reportType, selectedModel, dateRange });
    setReportDialogOpen(false);
  };

  const handleConfigureAutoRetrain = () => {
    setAutoRetrainDialogOpen(true);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'warning':
        return 'warning';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            Analytics & Insights
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Advanced analytics, drift detection, and ML operations monitoring
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button variant="outlined" startIcon={<Report />} onClick={() => setReportDialogOpen(true)}>
            Generate Report
          </Button>
          <Button variant="contained" startIcon={<AutoFixHigh />} onClick={handleConfigureAutoRetrain}>
            Auto-Retrain
          </Button>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Accuracy"
            value="94.2%"
            icon={<Assessment />}
            color="#4caf50"
            trend={2.1}
            subtitle="across all models"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Drift Alerts"
            value={mockDriftAlerts.length}
            icon={<Warning />}
            color="#ff9800"
            trend={-1}
            subtitle="active alerts"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Monthly Cost"
            value="$658.80"
            icon={<AttachMoney />}
            color="#2196f3"
            trend={8.5}
            subtitle="total spend"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Auto-Retrain Jobs"
            value={mockAutoRetrainJobs.length}
            icon={<AutoFixHigh />}
            color="#9c27b0"
            trend={1}
            subtitle="this week"
          />
        </Grid>
      </Grid>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={value} onChange={handleChange} aria-label="analytics tabs">
            <Tab icon={<Assessment />} label="Performance" />
            <Tab icon={<TrendingUp />} label="Drift Detection" />
            <Tab icon={<AutoFixHigh />} label="Auto-Retrain" />
            <Tab icon={<AttachMoney />} label="Cost Analytics" />
            <Tab icon={<Report />} label="Reports" />
          </Tabs>
        </Box>
        
        <TabPanel value={value} index={0}>
          <Typography variant="h6" gutterBottom>
            Model Performance Analytics
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor and analyze model performance metrics across all deployed models
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Performance Trends
                  </Typography>
                  <RechartsLineChart
                    data={[
                      { time: '00:00', accuracy: 94.2, latency: 45 },
                      { time: '04:00', accuracy: 94.5, latency: 42 },
                      { time: '08:00', accuracy: 93.8, latency: 48 },
                      { time: '12:00', accuracy: 94.1, latency: 44 },
                      { time: '16:00', accuracy: 94.3, latency: 46 },
                      { time: '20:00', accuracy: 94.0, latency: 47 },
                    ]}
                    dataKey="accuracy"
                    xAxisKey="time"
                    colors={['#4caf50', '#2196f3']}
                    width={600}
                    height={300}
                  />
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Model Performance
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Model</TableCell>
                          <TableCell>Accuracy</TableCell>
                          <TableCell>Status</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {mockPerformanceMetrics.map((model) => (
                          <TableRow key={model.id}>
                            <TableCell>
                              <Typography variant="subtitle2" fontWeight={600}>
                                {model.modelName}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {model.accuracy}%
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={model.status.toUpperCase()}
                                color={getStatusColor(model.status) as any}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={1}>
          <Typography variant="h6" gutterBottom>
            Drift Detection & Monitoring
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor data and concept drift to maintain model performance
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Drift Score Trends
                  </Typography>
                  <RechartsLineChart
                    data={[
                      { time: '00:00', driftScore: 0.08, threshold: 0.15 },
                      { time: '04:00', driftScore: 0.10, threshold: 0.15 },
                      { time: '08:00', driftScore: 0.12, threshold: 0.15 },
                      { time: '12:00', driftScore: 0.25, threshold: 0.15 },
                      { time: '16:00', driftScore: 0.18, threshold: 0.15 },
                      { time: '20:00', driftScore: 0.15, threshold: 0.15 },
                    ]}
                    dataKey="driftScore"
                    xAxisKey="time"
                    colors={['#f44336', '#ff9800']}
                    width={600}
                    height={300}
                  />
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Active Drift Alerts
                  </Typography>
                  <List>
                    {mockDriftAlerts.map((alert) => (
                      <ListItem key={alert.id} divider>
                        <ListItemIcon>
                          <Warning color={getSeverityColor(alert.severity) as any} />
                        </ListItemIcon>
                        <ListItemText
                          primary={alert.modelName}
                          secondary={`${alert.driftType} • ${(alert.driftScore * 100).toFixed(1)}%`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={2}>
          <Typography variant="h6" gutterBottom>
            Auto-Retrain Management
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Configure and monitor automatic model retraining based on performance and drift
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Auto-Retrain Jobs
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Model</TableCell>
                          <TableCell>Trigger</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Priority</TableCell>
                          <TableCell>Schedule</TableCell>
                          <TableCell>Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {mockAutoRetrainJobs.map((job) => (
                          <TableRow key={job.id}>
                            <TableCell>
                              <Typography variant="subtitle2" fontWeight={600}>
                                {job.modelName}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {job.trigger}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={job.status.toUpperCase()}
                                color={getStatusColor(job.status) as any}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={job.priority.toUpperCase()}
                                color={getPriorityColor(job.priority) as any}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {job.scheduledAt || job.startedAt || job.completedAt}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Box display="flex" gap={1}>
                                <IconButton size="small">
                                  <Visibility />
                                </IconButton>
                                <IconButton size="small">
                                  <Edit />
                                </IconButton>
                                <IconButton size="small">
                                  <Delete />
                                </IconButton>
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Auto-Retrain Configuration
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Drift Detection"
                        secondary="Enabled for all models"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Performance Monitoring"
                        secondary="Accuracy threshold: 90%"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Scheduled Retraining"
                        secondary="Weekly for production models"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Warning color="warning" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Resource Limits"
                        secondary="Max 2 concurrent jobs"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={3}>
          <Typography variant="h6" gutterBottom>
            Cost Analytics
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Track and optimize costs across all ML services and resources
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Cost Breakdown by Service
                  </Typography>
                  <RechartsBarChart
                    data={mockCostMetrics}
                    dataKey="cost"
                    xAxisKey="service"
                    colors={['#2196f3']}
                    width={600}
                    height={300}
                  />
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Cost Summary
                  </Typography>
                  <List>
                    {mockCostMetrics.map((metric, index) => (
                      <ListItem key={index} divider>
                        <ListItemText
                          primary={metric.service}
                          secondary={`$${metric.cost} • ${metric.usage}`}
                        />
                        <Typography variant="body2" color={metric.trend > 0 ? 'error' : 'success'}>
                          {metric.trend > 0 ? '+' : ''}{metric.trend}%
                        </Typography>
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={4}>
          <Typography variant="h6" gutterBottom>
            Reports & Insights
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Generate comprehensive reports and gain insights into your ML operations
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Available Reports
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <Assessment />
                      </ListItemIcon>
                      <ListItemText
                        primary="Performance Report"
                        secondary="Model accuracy, latency, and throughput analysis"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <TrendingUp />
                      </ListItemIcon>
                      <ListItemText
                        primary="Drift Analysis"
                        secondary="Data and concept drift detection summary"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <AttachMoney />
                      </ListItemIcon>
                      <ListItemText
                        primary="Cost Report"
                        secondary="Resource usage and cost optimization insights"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Timeline />
                      </ListItemIcon>
                      <ListItemText
                        primary="Training Report"
                        secondary="Model training history and performance trends"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Reports
                  </Typography>
                  <List>
                    <ListItem divider>
                      <ListItemText
                        primary="Weekly Performance Summary"
                        secondary="Generated 2 hours ago"
                      />
                      <Button size="small" startIcon={<Download />}>
                        Download
                      </Button>
                    </ListItem>
                    <ListItem divider>
                      <ListItemText
                        primary="Monthly Cost Analysis"
                        secondary="Generated 1 day ago"
                      />
                      <Button size="small" startIcon={<Download />}>
                        Download
                      </Button>
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Drift Detection Report"
                        secondary="Generated 3 days ago"
                      />
                      <Button size="small" startIcon={<Download />}>
                        Download
                      </Button>
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

      {/* Generate Report Dialog */}
      <Dialog open={reportDialogOpen} onClose={() => setReportDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Generate Report</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Report Type</InputLabel>
            <Select value={reportType} onChange={(e) => setReportType(e.target.value)}>
              <MenuItem value="performance">Performance Report</MenuItem>
              <MenuItem value="drift">Drift Analysis</MenuItem>
              <MenuItem value="cost">Cost Report</MenuItem>
              <MenuItem value="training">Training Report</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Model</InputLabel>
            <Select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              <MenuItem value="all">All Models</MenuItem>
              <MenuItem value="bert">BERT Security</MenuItem>
              <MenuItem value="distilbert">DistilBERT</MenuItem>
              <MenuItem value="roberta">RoBERTa Large</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Date Range</InputLabel>
            <Select value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
              <MenuItem value="1d">Last 24 Hours</MenuItem>
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
              <MenuItem value="90d">Last 90 Days</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReportDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleGenerateReport}>
            Generate Report
          </Button>
        </DialogActions>
      </Dialog>

      {/* Auto-Retrain Configuration Dialog */}
      <Dialog open={autoRetrainDialogOpen} onClose={() => setAutoRetrainDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Configure Auto-Retrain</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Model</InputLabel>
            <Select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              <MenuItem value="bert">BERT Security</MenuItem>
              <MenuItem value="distilbert">DistilBERT</MenuItem>
              <MenuItem value="roberta">RoBERTa Large</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Trigger Type</InputLabel>
            <Select>
              <MenuItem value="drift">Drift Detection</MenuItem>
              <MenuItem value="performance">Performance Drop</MenuItem>
              <MenuItem value="scheduled">Scheduled</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Priority</InputLabel>
            <Select>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="low">Low</MenuItem>
            </Select>
          </FormControl>
          <FormGroup>
            <FormControlLabel control={<Switch defaultChecked />} label="Enable Auto-Retrain" />
            <FormControlLabel control={<Switch />} label="Send Notifications" />
          </FormGroup>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAutoRetrainDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setAutoRetrainDialogOpen(false)}>
            Save Configuration
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Analytics;