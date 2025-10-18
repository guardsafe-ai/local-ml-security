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
  TreeView,
  TreeItem,
} from '@mui/material';
import {
  Monitor,
  Speed,
  Memory,
  Storage,
  NetworkCheck,
  Assessment,
  Timeline,
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
  Lock,
  LockOpen,
  VisibilityOff,
  Person,
  Email,
  Phone,
  CreditCard,
  LocationOn,
  Fingerprint,
  Gavel,
  Policy,
  Compliance,
  Audit,
  DataObject,
  CloudUpload,
  CloudDownload,
  Computer,
  Router,
  Database,
  Security,
  BugReport,
  Code,
  Terminal,
  Cloud,
  Wifi,
  WifiOff,
  SignalCellular4Bar,
  SignalCellularOff,
  BatteryFull,
  BatteryAlert,
  Thermostat,
  AcUnit,
  Power,
  PowerOff,
  RestartAlt,
  Update,
  Sync,
  SyncProblem,
  CloudSync,
  CloudDone,
  CloudOff,
  CloudQueue,
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
      id={`monitoring-tabpanel-${index}`}
      aria-labelledby={`monitoring-tab-${index}`}
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
const mockServices = [
  {
    id: '1',
    name: 'Model API Service',
    status: 'healthy',
    uptime: 99.9,
    responseTime: 45,
    cpu: 25,
    memory: 60,
    lastCheck: '2024-01-20 14:30:00',
    version: 'v2.1.3',
    instances: 3,
  },
  {
    id: '2',
    name: 'Training Service',
    status: 'warning',
    uptime: 98.5,
    responseTime: 120,
    cpu: 85,
    memory: 90,
    lastCheck: '2024-01-20 14:29:00',
    version: 'v1.8.2',
    instances: 2,
  },
  {
    id: '3',
    name: 'Data Service',
    status: 'healthy',
    uptime: 99.8,
    responseTime: 30,
    cpu: 15,
    memory: 40,
    lastCheck: '2024-01-20 14:30:00',
    version: 'v3.0.1',
    instances: 4,
  },
  {
    id: '4',
    name: 'Analytics Service',
    status: 'error',
    uptime: 95.2,
    responseTime: 500,
    cpu: 95,
    memory: 95,
    lastCheck: '2024-01-20 14:25:00',
    version: 'v1.5.0',
    instances: 1,
  },
];

const mockAlerts = [
  {
    id: '1',
    severity: 'high',
    service: 'Analytics Service',
    message: 'High CPU usage detected',
    timestamp: '2024-01-20 14:25:00',
    status: 'active',
    acknowledged: false,
  },
  {
    id: '2',
    severity: 'medium',
    service: 'Training Service',
    message: 'Memory usage above threshold',
    timestamp: '2024-01-20 14:20:00',
    status: 'active',
    acknowledged: true,
  },
  {
    id: '3',
    severity: 'low',
    service: 'Model API Service',
    message: 'Response time slightly elevated',
    timestamp: '2024-01-20 14:15:00',
    status: 'resolved',
    acknowledged: true,
  },
];

const mockResourceUsage = [
  {
    timestamp: '14:00',
    cpu: 45,
    memory: 60,
    disk: 25,
    network: 30,
  },
  {
    timestamp: '14:05',
    cpu: 50,
    memory: 65,
    disk: 26,
    network: 35,
  },
  {
    timestamp: '14:10',
    cpu: 55,
    memory: 70,
    disk: 27,
    network: 40,
  },
  {
    timestamp: '14:15',
    cpu: 60,
    memory: 75,
    disk: 28,
    network: 45,
  },
  {
    timestamp: '14:20',
    cpu: 65,
    memory: 80,
    disk: 29,
    network: 50,
  },
  {
    timestamp: '14:25',
    cpu: 70,
    memory: 85,
    disk: 30,
    network: 55,
  },
  {
    timestamp: '14:30',
    cpu: 75,
    memory: 90,
    disk: 31,
    network: 60,
  },
];

const mockLogs = [
  {
    id: '1',
    timestamp: '2024-01-20 14:30:15',
    level: 'ERROR',
    service: 'Analytics Service',
    message: 'Failed to process batch request',
    details: 'Connection timeout to database',
  },
  {
    id: '2',
    timestamp: '2024-01-20 14:29:45',
    level: 'WARN',
    service: 'Training Service',
    message: 'Memory usage high',
    details: 'Current usage: 90%, threshold: 85%',
  },
  {
    id: '3',
    timestamp: '2024-01-20 14:29:30',
    level: 'INFO',
    service: 'Model API Service',
    message: 'Model loaded successfully',
    details: 'BERT Security v2.1.3 loaded in 2.3s',
  },
  {
    id: '4',
    timestamp: '2024-01-20 14:29:15',
    level: 'DEBUG',
    service: 'Data Service',
    message: 'Processing data batch',
    details: 'Batch size: 1000 records',
  },
];

const mockTracing = [
  {
    id: '1',
    traceId: 'abc123def456',
    service: 'Model API Service',
    operation: 'predict',
    duration: 45,
    status: 'success',
    timestamp: '2024-01-20 14:30:00',
    spans: 3,
  },
  {
    id: '2',
    traceId: 'xyz789uvw012',
    service: 'Training Service',
    operation: 'train_model',
    duration: 1200,
    status: 'success',
    timestamp: '2024-01-20 14:25:00',
    spans: 8,
  },
  {
    id: '3',
    traceId: 'mno345pqr678',
    service: 'Analytics Service',
    operation: 'analyze',
    duration: 500,
    status: 'error',
    timestamp: '2024-01-20 14:20:00',
    spans: 5,
  },
];

const Monitoring: React.FC = () => {
  const [value, setValue] = useState(0);
  const [alertDialogOpen, setAlertDialogOpen] = useState(false);
  const [logDialogOpen, setLogDialogOpen] = useState(false);
  const [selectedService, setSelectedService] = useState('');
  const [alertMessage, setAlertMessage] = useState('');
  const [alertSeverity, setAlertSeverity] = useState('');

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const handleCreateAlert = () => {
    console.log('Creating alert:', { selectedService, alertMessage, alertSeverity });
    setAlertDialogOpen(false);
    setSelectedService('');
    setAlertMessage('');
    setAlertSeverity('');
  };

  const handleViewLogs = () => {
    console.log('Viewing logs');
    setLogDialogOpen(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      case 'active':
        return 'error';
      case 'resolved':
        return 'success';
      case 'success':
        return 'success';
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

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR':
        return 'error';
      case 'WARN':
        return 'warning';
      case 'INFO':
        return 'info';
      case 'DEBUG':
        return 'default';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            System Monitoring
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Real-time system health, resource monitoring, and observability
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button variant="outlined" startIcon={<BugReport />} onClick={() => setAlertDialogOpen(true)}>
            Create Alert
          </Button>
          <Button variant="contained" startIcon={<Refresh />}>
            Refresh All
          </Button>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="System Health"
            value="87%"
            icon={<Monitor />}
            color="#4caf50"
            trend={2.1}
            subtitle="overall status"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Alerts"
            value={mockAlerts.filter(a => a.status === 'active').length}
            icon={<Warning />}
            color="#f44336"
            trend={-1}
            subtitle="critical issues"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="CPU Usage"
            value="65%"
            icon={<Speed />}
            color="#ff9800"
            trend={5.2}
            subtitle="average load"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Memory Usage"
            value="72%"
            icon={<Memory />}
            color="#2196f3"
            trend={3.8}
            subtitle="total memory"
          />
        </Grid>
      </Grid>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={value} onChange={handleChange} aria-label="monitoring tabs">
            <Tab icon={<Monitor />} label="Service Health" />
            <Tab icon={<Speed />} label="Resources" />
            <Tab icon={<Warning />} label="Alerts" />
            <Tab icon={<Code />} label="Logs" />
            <Tab icon={<Timeline />} label="Tracing" />
          </Tabs>
        </Box>
        
        <TabPanel value={value} index={0}>
          <Typography variant="h6" gutterBottom>
            Service Health Overview
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor the health and performance of all system services
          </Typography>

          <Grid container spacing={3}>
            {mockServices.map((service) => (
              <Grid item xs={12} md={6} lg={3} key={service.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {service.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          v{service.version} • {service.instances} instances
                        </Typography>
                      </Box>
                      <Chip
                        label={service.status.toUpperCase()}
                        color={getStatusColor(service.status) as any}
                        size="small"
                      />
                    </Box>

                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {service.uptime}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Uptime
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {service.responseTime}ms
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Response
                        </Typography>
                      </Box>
                    </Box>

                    <Box mb={2}>
                      <Box display="flex" justifyContent="space-between" mb={1}>
                        <Typography variant="body2" color="text.secondary">
                          CPU Usage
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {service.cpu}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={service.cpu}
                        color={service.cpu > 80 ? 'error' : service.cpu > 60 ? 'warning' : 'success'}
                      />
                    </Box>

                    <Box mb={2}>
                      <Box display="flex" justifyContent="space-between" mb={1}>
                        <Typography variant="body2" color="text.secondary">
                          Memory Usage
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {service.memory}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={service.memory}
                        color={service.memory > 80 ? 'error' : service.memory > 60 ? 'warning' : 'success'}
                      />
                    </Box>

                    <Typography variant="caption" color="text.secondary" mb={2}>
                      Last check: {service.lastCheck}
                    </Typography>

                    <Box display="flex" gap={1} mt={2}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Visibility />}
                        fullWidth
                      >
                        Details
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<RestartAlt />}
                        fullWidth
                      >
                        Restart
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={1}>
          <Typography variant="h6" gutterBottom>
            Resource Usage & Performance
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor system resources and performance metrics in real-time
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Resource Usage Over Time
                  </Typography>
                  <RechartsLineChart
                    data={mockResourceUsage}
                    dataKey="cpu"
                    xAxisKey="timestamp"
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
                    Current Usage
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <Speed />
                      </ListItemIcon>
                      <ListItemText
                        primary="CPU Usage"
                        secondary="75%"
                      />
                      <LinearProgress
                        variant="determinate"
                        value={75}
                        color="warning"
                        sx={{ width: 60 }}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Memory />
                      </ListItemIcon>
                      <ListItemText
                        primary="Memory Usage"
                        secondary="90%"
                      />
                      <LinearProgress
                        variant="determinate"
                        value={90}
                        color="error"
                        sx={{ width: 60 }}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Storage />
                      </ListItemIcon>
                      <ListItemText
                        primary="Disk Usage"
                        secondary="31%"
                      />
                      <LinearProgress
                        variant="determinate"
                        value={31}
                        color="success"
                        sx={{ width: 60 }}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <NetworkCheck />
                      </ListItemIcon>
                      <ListItemText
                        primary="Network Usage"
                        secondary="60%"
                      />
                      <LinearProgress
                        variant="determinate"
                        value={60}
                        color="info"
                        sx={{ width: 60 }}
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={2}>
          <Typography variant="h6" gutterBottom>
            System Alerts & Notifications
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Manage system alerts and notifications
          </Typography>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Severity</TableCell>
                  <TableCell>Service</TableCell>
                  <TableCell>Message</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Acknowledged</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {mockAlerts.map((alert) => (
                  <TableRow key={alert.id}>
                    <TableCell>
                      <Chip
                        label={alert.severity.toUpperCase()}
                        color={getSeverityColor(alert.severity) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="subtitle2" fontWeight={600}>
                        {alert.service}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {alert.message}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {alert.timestamp}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={alert.status.toUpperCase()}
                        color={getStatusColor(alert.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={alert.acknowledged ? 'YES' : 'NO'}
                        color={alert.acknowledged ? 'success' : 'warning'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <IconButton size="small">
                          <Visibility />
                        </IconButton>
                        <IconButton size="small">
                          <CheckCircle />
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
        </TabPanel>
        
        <TabPanel value={value} index={3}>
          <Typography variant="h6" gutterBottom>
            System Logs
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            View and search system logs in real-time
          </Typography>

          <Box display="flex" gap={2} mb={3}>
            <TextField
              placeholder="Search logs..."
              size="small"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Level</InputLabel>
              <Select>
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="error">ERROR</MenuItem>
                <MenuItem value="warn">WARN</MenuItem>
                <MenuItem value="info">INFO</MenuItem>
                <MenuItem value="debug">DEBUG</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Service</InputLabel>
              <Select>
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="model-api">Model API</MenuItem>
                <MenuItem value="training">Training</MenuItem>
                <MenuItem value="data">Data</MenuItem>
                <MenuItem value="analytics">Analytics</MenuItem>
              </Select>
            </FormControl>
            <Button variant="outlined" startIcon={<Download />}>
              Export
            </Button>
          </Box>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Level</TableCell>
                  <TableCell>Service</TableCell>
                  <TableCell>Message</TableCell>
                  <TableCell>Details</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {mockLogs.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell>
                      <Typography variant="body2">
                        {log.timestamp}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={log.level}
                        color={getLogLevelColor(log.level) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {log.service}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {log.message}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {log.details}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <IconButton size="small">
                        <Visibility />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
        
        <TabPanel value={value} index={4}>
          <Typography variant="h6" gutterBottom>
            Distributed Tracing
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor request flows and performance across services
          </Typography>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Trace ID</TableCell>
                  <TableCell>Service</TableCell>
                  <TableCell>Operation</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Spans</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {mockTracing.map((trace) => (
                  <TableRow key={trace.id}>
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {trace.traceId}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {trace.service}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {trace.operation}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {trace.duration}ms
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={trace.status.toUpperCase()}
                        color={getStatusColor(trace.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {trace.spans}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {trace.timestamp}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <IconButton size="small">
                          <Visibility />
                        </IconButton>
                        <IconButton size="small">
                          <Timeline />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
      </Paper>

      {/* Alert Dialog */}
      <Dialog open={alertDialogOpen} onClose={() => setAlertDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create System Alert</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Service</InputLabel>
            <Select value={selectedService} onChange={(e) => setSelectedService(e.target.value)}>
              <MenuItem value="model-api">Model API Service</MenuItem>
              <MenuItem value="training">Training Service</MenuItem>
              <MenuItem value="data">Data Service</MenuItem>
              <MenuItem value="analytics">Analytics Service</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Severity</InputLabel>
            <Select value={alertSeverity} onChange={(e) => setAlertSeverity(e.target.value)}>
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="high">High</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Alert Message"
            value={alertMessage}
            onChange={(e) => setAlertMessage(e.target.value)}
            margin="normal"
            multiline
            rows={3}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAlertDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleCreateAlert}>
            Create Alert
          </Button>
        </DialogActions>
      </Dialog>

      {/* Logs Dialog */}
      <Dialog open={logDialogOpen} onClose={() => setLogDialogOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle>System Logs</DialogTitle>
        <DialogContent>
          <Box sx={{ height: 400, overflow: 'auto' }}>
            <Typography variant="body2" fontFamily="monospace" sx={{ whiteSpace: 'pre-wrap' }}>
              {mockLogs.map(log => 
                `[${log.timestamp}] ${log.level} ${log.service}: ${log.message}\n${log.details}\n`
              ).join('\n')}
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLogDialogOpen(false)}>Close</Button>
          <Button variant="contained" onClick={handleViewLogs}>
            Export Logs
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Monitoring;