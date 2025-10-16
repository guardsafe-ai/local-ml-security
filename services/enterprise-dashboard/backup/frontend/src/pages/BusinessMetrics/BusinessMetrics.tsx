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
  Autocomplete,
  DatePicker,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AttachMoney,
  Assessment,
  Timeline,
  Speed,
  Memory,
  Business,
  Analytics,
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
  AccountBalance,
  MonetizationOn,
  TrendingFlat,
  Compare,
  Calculate,
  Dashboard,
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
      id={`business-tabpanel-${index}`}
      aria-labelledby={`business-tab-${index}`}
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
const mockKPIs = [
  {
    id: '1',
    name: 'Model Accuracy',
    value: 94.2,
    target: 95.0,
    unit: '%',
    trend: 2.1,
    status: 'on-track',
    category: 'Performance',
    lastUpdated: '2024-01-20 14:30:00',
  },
  {
    id: '2',
    name: 'Cost per Prediction',
    value: 0.045,
    target: 0.040,
    unit: '$',
    trend: -8.5,
    status: 'exceeding',
    category: 'Cost',
    lastUpdated: '2024-01-20 13:45:00',
  },
  {
    id: '3',
    name: 'ROI',
    value: 285.5,
    target: 250.0,
    unit: '%',
    trend: 15.2,
    status: 'exceeding',
    category: 'Business',
    lastUpdated: '2024-01-20 12:15:00',
  },
  {
    id: '4',
    name: 'Model Uptime',
    value: 99.8,
    target: 99.5,
    unit: '%',
    trend: 0.3,
    status: 'on-track',
    category: 'Reliability',
    lastUpdated: '2024-01-20 14:30:00',
  },
];

const mockCostBreakdown = [
  {
    category: 'Infrastructure',
    amount: 1250.50,
    percentage: 35.2,
    trend: 5.2,
    subcategories: [
      { name: 'Compute', amount: 750.30, percentage: 21.1 },
      { name: 'Storage', amount: 300.20, percentage: 8.4 },
      { name: 'Network', amount: 200.00, percentage: 5.6 },
    ],
  },
  {
    category: 'Development',
    amount: 980.75,
    percentage: 27.6,
    trend: -2.1,
    subcategories: [
      { name: 'Training', amount: 600.25, percentage: 16.9 },
      { name: 'Testing', amount: 200.50, percentage: 5.6 },
      { name: 'Deployment', amount: 180.00, percentage: 5.1 },
    ],
  },
  {
    category: 'Operations',
    amount: 890.25,
    percentage: 25.1,
    trend: 12.3,
    subcategories: [
      { name: 'Monitoring', amount: 400.00, percentage: 11.3 },
      { name: 'Maintenance', amount: 300.25, percentage: 8.4 },
      { name: 'Support', amount: 190.00, percentage: 5.3 },
    ],
  },
  {
    category: 'Data',
    amount: 425.50,
    percentage: 12.0,
    trend: 8.7,
    subcategories: [
      { name: 'Storage', amount: 200.00, percentage: 5.6 },
      { name: 'Processing', amount: 150.50, percentage: 4.2 },
      { name: 'Quality', amount: 75.00, percentage: 2.1 },
    ],
  },
];

const mockROIAnalysis = [
  {
    model: 'BERT Security',
    investment: 15000,
    returns: 45000,
    roi: 200.0,
    paybackPeriod: 3.2,
    npv: 28500,
    irr: 45.2,
  },
  {
    model: 'DistilBERT',
    investment: 8000,
    returns: 22000,
    roi: 175.0,
    paybackPeriod: 2.8,
    npv: 13500,
    irr: 38.5,
  },
  {
    model: 'RoBERTa Large',
    investment: 25000,
    returns: 65000,
    roi: 160.0,
    paybackPeriod: 4.1,
    npv: 37500,
    irr: 32.8,
  },
];

const mockCustomMetrics = [
  {
    id: '1',
    name: 'Customer Satisfaction Score',
    value: 4.7,
    unit: '/5',
    target: 4.5,
    trend: 0.2,
    category: 'Customer',
    description: 'Average customer satisfaction rating',
    lastUpdated: '2024-01-20 14:30:00',
  },
  {
    id: '2',
    name: 'Time to Market',
    value: 12.5,
    unit: 'days',
    target: 15.0,
    trend: -2.5,
    category: 'Efficiency',
    description: 'Average time to deploy new models',
    lastUpdated: '2024-01-20 13:45:00',
  },
  {
    id: '3',
    name: 'Data Quality Index',
    value: 92.3,
    unit: '%',
    target: 90.0,
    trend: 3.2,
    category: 'Quality',
    description: 'Overall data quality score',
    lastUpdated: '2024-01-20 12:15:00',
  },
];

const BusinessMetrics: React.FC = () => {
  const [value, setValue] = useState(0);
  const [customMetricDialogOpen, setCustomMetricDialogOpen] = useState(false);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState('');
  const [metricName, setMetricName] = useState('');
  const [metricValue, setMetricValue] = useState('');
  const [metricTarget, setMetricTarget] = useState('');
  const [metricUnit, setMetricUnit] = useState('');
  const [metricCategory, setMetricCategory] = useState('');

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const handleCreateCustomMetric = () => {
    console.log('Creating custom metric:', { metricName, metricValue, metricTarget, metricUnit, metricCategory });
    setCustomMetricDialogOpen(false);
    setMetricName('');
    setMetricValue('');
    setMetricTarget('');
    setMetricUnit('');
    setMetricCategory('');
  };

  const handleGenerateReport = () => {
    console.log('Generating business report');
    setReportDialogOpen(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'exceeding':
        return 'success';
      case 'on-track':
        return 'info';
      case 'behind':
        return 'warning';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getTrendIcon = (trend: number) => {
    if (trend > 0) return <TrendingUp color="success" />;
    if (trend < 0) return <TrendingDown color="error" />;
    return <TrendingFlat color="info" />;
  };

  const getTrendColor = (trend: number) => {
    if (trend > 0) return 'success';
    if (trend < 0) return 'error';
    return 'info';
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            Business Metrics
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Enterprise KPIs, cost analysis, and ROI tracking
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button variant="outlined" startIcon={<Report />} onClick={() => setReportDialogOpen(true)}>
            Generate Report
          </Button>
          <Button variant="contained" startIcon={<Add />} onClick={() => setCustomMetricDialogOpen(true)}>
            Add Metric
          </Button>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total ROI"
            value="285.5%"
            icon={<TrendingUp />}
            color="#4caf50"
            trend={15.2}
            subtitle="across all models"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Monthly Cost"
            value="$3,546"
            icon={<AttachMoney />}
            color="#f44336"
            trend={8.5}
            subtitle="total spend"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Cost per Prediction"
            value="$0.045"
            icon={<MonetizationOn />}
            color="#ff9800"
            trend={-8.5}
            subtitle="average cost"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active KPIs"
            value={mockKPIs.length}
            icon={<Assessment />}
            color="#2196f3"
            trend={2}
            subtitle="tracked metrics"
          />
        </Grid>
      </Grid>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={value} onChange={handleChange} aria-label="business metrics tabs">
            <Tab icon={<Assessment />} label="KPIs" />
            <Tab icon={<AttachMoney />} label="Cost Analysis" />
            <Tab icon={<TrendingUp />} label="ROI Analysis" />
            <Tab icon={<Add />} label="Custom Metrics" />
            <Tab icon={<Report />} label="Reports" />
          </Tabs>
        </Box>
        
        <TabPanel value={value} index={0}>
          <Typography variant="h6" gutterBottom>
            Key Performance Indicators
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor critical business metrics and track progress against targets
          </Typography>

          <Grid container spacing={3}>
            {mockKPIs.map((kpi) => (
              <Grid item xs={12} md={6} lg={3} key={kpi.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {kpi.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {kpi.category}
                        </Typography>
                      </Box>
                      <Box display="flex" alignItems="center" gap={1}>
                        {getTrendIcon(kpi.trend)}
                        <Chip
                          label={kpi.status.toUpperCase()}
                          color={getStatusColor(kpi.status) as any}
                          size="small"
                        />
                      </Box>
                    </Box>

                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Typography variant="h4" color="primary">
                        {kpi.value}{kpi.unit}
                      </Typography>
                      <Typography variant="body2" color={getTrendColor(kpi.trend)}>
                        {kpi.trend > 0 ? '+' : ''}{kpi.trend}%
                      </Typography>
                    </Box>

                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Typography variant="body2" color="text.secondary">
                        Target: {kpi.target}{kpi.unit}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {((kpi.value / kpi.target) * 100).toFixed(1)}%
                      </Typography>
                    </Box>

                    <LinearProgress
                      variant="determinate"
                      value={Math.min((kpi.value / kpi.target) * 100, 100)}
                      color={kpi.value >= kpi.target ? 'success' : 'primary'}
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={1}>
          <Typography variant="h6" gutterBottom>
            Cost Analysis & Breakdown
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Detailed cost analysis and optimization opportunities
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Cost Breakdown by Category
                  </Typography>
                  <RechartsPieChart
                    data={mockCostBreakdown}
                    dataKey="amount"
                    nameKey="category"
                    colors={['#2196f3', '#4caf50', '#ff9800', '#f44336']}
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
                    {mockCostBreakdown.map((category, index) => (
                      <ListItem key={index} divider>
                        <ListItemText
                          primary={category.category}
                          secondary={`$${category.amount.toLocaleString()} (${category.percentage}%)`}
                        />
                        <Typography variant="body2" color={getTrendColor(category.trend)}>
                          {category.trend > 0 ? '+' : ''}{category.trend}%
                        </Typography>
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Grid container spacing={3} mt={2}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Detailed Cost Breakdown
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Category</TableCell>
                          <TableCell>Subcategory</TableCell>
                          <TableCell>Amount</TableCell>
                          <TableCell>Percentage</TableCell>
                          <TableCell>Trend</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {mockCostBreakdown.map((category) =>
                          category.subcategories.map((sub, index) => (
                            <TableRow key={`${category.category}-${index}`}>
                              <TableCell>
                                {index === 0 && (
                                  <Typography variant="subtitle2" fontWeight={600}>
                                    {category.category}
                                  </Typography>
                                )}
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2">
                                  {sub.name}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2">
                                  ${sub.amount.toLocaleString()}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2">
                                  {sub.percentage}%
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2" color={getTrendColor(category.trend)}>
                                  {category.trend > 0 ? '+' : ''}{category.trend}%
                                </Typography>
                              </TableCell>
                            </TableRow>
                          ))
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={2}>
          <Typography variant="h6" gutterBottom>
            ROI Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Return on investment analysis for ML models and initiatives
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ROI by Model
                  </Typography>
                  <RechartsBarChart
                    data={mockROIAnalysis}
                    dataKey="roi"
                    xAxisKey="model"
                    colors={['#4caf50']}
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
                    ROI Summary
                  </Typography>
                  <List>
                    {mockROIAnalysis.map((model, index) => (
                      <ListItem key={index} divider>
                        <ListItemText
                          primary={model.model}
                          secondary={`ROI: ${model.roi}%`}
                        />
                        <Typography variant="body2" color="success">
                          ${model.npv.toLocaleString()}
                        </Typography>
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Grid container spacing={3} mt={2}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Detailed ROI Analysis
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Model</TableCell>
                          <TableCell>Investment</TableCell>
                          <TableCell>Returns</TableCell>
                          <TableCell>ROI</TableCell>
                          <TableCell>Payback Period</TableCell>
                          <TableCell>NPV</TableCell>
                          <TableCell>IRR</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {mockROIAnalysis.map((model, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Typography variant="subtitle2" fontWeight={600}>
                                {model.model}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                ${model.investment.toLocaleString()}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                ${model.returns.toLocaleString()}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="success">
                                {model.roi}%
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {model.paybackPeriod} months
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="success">
                                ${model.npv.toLocaleString()}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="success">
                                {model.irr}%
                              </Typography>
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
        
        <TabPanel value={value} index={3}>
          <Typography variant="h6" gutterBottom>
            Custom Metrics
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Create and track custom business metrics specific to your organization
          </Typography>

          <Grid container spacing={3}>
            {mockCustomMetrics.map((metric) => (
              <Grid item xs={12} md={6} lg={4} key={metric.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {metric.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {metric.category}
                        </Typography>
                      </Box>
                      <Box display="flex" gap={1}>
                        <IconButton size="small">
                          <Edit />
                        </IconButton>
                        <IconButton size="small">
                          <Delete />
                        </IconButton>
                      </Box>
                    </Box>

                    <Typography variant="body2" color="text.secondary" mb={2}>
                      {metric.description}
                    </Typography>

                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Typography variant="h4" color="primary">
                        {metric.value}{metric.unit}
                      </Typography>
                      <Typography variant="body2" color={getTrendColor(metric.trend)}>
                        {metric.trend > 0 ? '+' : ''}{metric.trend}%
                      </Typography>
                    </Box>

                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Typography variant="body2" color="text.secondary">
                        Target: {metric.target}{metric.unit}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {((metric.value / metric.target) * 100).toFixed(1)}%
                      </Typography>
                    </Box>

                    <LinearProgress
                      variant="determinate"
                      value={Math.min((metric.value / metric.target) * 100, 100)}
                      color={metric.value >= metric.target ? 'success' : 'primary'}
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={4}>
          <Typography variant="h6" gutterBottom>
            Business Reports
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Generate comprehensive business reports and insights
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
                        primary="KPI Dashboard"
                        secondary="Comprehensive KPI tracking and analysis"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <AttachMoney />
                      </ListItemIcon>
                      <ListItemText
                        primary="Cost Analysis Report"
                        secondary="Detailed cost breakdown and optimization"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <TrendingUp />
                      </ListItemIcon>
                      <ListItemText
                        primary="ROI Analysis"
                        secondary="Return on investment analysis"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Business />
                      </ListItemIcon>
                      <ListItemText
                        primary="Executive Summary"
                        secondary="High-level business impact summary"
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
                        primary="Monthly KPI Report"
                        secondary="Generated 2 hours ago"
                      />
                      <Button size="small" startIcon={<Download />}>
                        Download
                      </Button>
                    </ListItem>
                    <ListItem divider>
                      <ListItemText
                        primary="Q4 Cost Analysis"
                        secondary="Generated 1 day ago"
                      />
                      <Button size="small" startIcon={<Download />}>
                        Download
                      </Button>
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="ROI Summary"
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

      {/* Custom Metric Dialog */}
      <Dialog open={customMetricDialogOpen} onClose={() => setCustomMetricDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Custom Metric</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Metric Name"
            value={metricName}
            onChange={(e) => setMetricName(e.target.value)}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Current Value"
            value={metricValue}
            onChange={(e) => setMetricValue(e.target.value)}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Target Value"
            value={metricTarget}
            onChange={(e) => setMetricTarget(e.target.value)}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Unit"
            value={metricUnit}
            onChange={(e) => setMetricUnit(e.target.value)}
            margin="normal"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Category</InputLabel>
            <Select value={metricCategory} onChange={(e) => setMetricCategory(e.target.value)}>
              <MenuItem value="Performance">Performance</MenuItem>
              <MenuItem value="Cost">Cost</MenuItem>
              <MenuItem value="Quality">Quality</MenuItem>
              <MenuItem value="Efficiency">Efficiency</MenuItem>
              <MenuItem value="Customer">Customer</MenuItem>
              <MenuItem value="Business">Business</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCustomMetricDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleCreateCustomMetric}>
            Create Metric
          </Button>
        </DialogActions>
      </Dialog>

      {/* Generate Report Dialog */}
      <Dialog open={reportDialogOpen} onClose={() => setReportDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Generate Business Report</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Report Type</InputLabel>
            <Select>
              <MenuItem value="kpi">KPI Dashboard</MenuItem>
              <MenuItem value="cost">Cost Analysis</MenuItem>
              <MenuItem value="roi">ROI Analysis</MenuItem>
              <MenuItem value="executive">Executive Summary</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Time Period</InputLabel>
            <Select>
              <MenuItem value="monthly">Monthly</MenuItem>
              <MenuItem value="quarterly">Quarterly</MenuItem>
              <MenuItem value="yearly">Yearly</MenuItem>
              <MenuItem value="custom">Custom Range</MenuItem>
            </Select>
          </FormControl>
          <FormGroup>
            <FormControlLabel control={<Checkbox defaultChecked />} label="Include Charts" />
            <FormControlLabel control={<Checkbox defaultChecked />} label="Include Recommendations" />
            <FormControlLabel control={<Checkbox />} label="Include Raw Data" />
          </FormGroup>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReportDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleGenerateReport}>
            Generate Report
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default BusinessMetrics;