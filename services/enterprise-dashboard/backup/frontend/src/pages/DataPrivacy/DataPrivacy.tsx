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
  Security,
  PrivacyTip,
  Shield,
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
  Storage,
  CloudUpload,
  CloudDownload,
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
      id={`privacy-tabpanel-${index}`}
      aria-labelledby={`privacy-tab-${index}`}
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
const mockDataClassification = [
  {
    id: '1',
    name: 'Personal Identifiable Information',
    category: 'PII',
    sensitivity: 'high',
    count: 1250,
    lastScanned: '2024-01-20 14:30:00',
    status: 'protected',
    description: 'Names, emails, phone numbers, addresses',
  },
  {
    id: '2',
    name: 'Financial Information',
    category: 'Financial',
    sensitivity: 'critical',
    count: 890,
    lastScanned: '2024-01-20 13:45:00',
    status: 'encrypted',
    description: 'Credit card numbers, bank accounts, SSNs',
  },
  {
    id: '3',
    name: 'Health Information',
    category: 'Health',
    sensitivity: 'critical',
    count: 450,
    lastScanned: '2024-01-20 12:15:00',
    status: 'encrypted',
    description: 'Medical records, health conditions, treatments',
  },
  {
    id: '4',
    name: 'Business Information',
    category: 'Business',
    sensitivity: 'medium',
    count: 2100,
    lastScanned: '2024-01-20 11:30:00',
    status: 'protected',
    description: 'Internal documents, strategies, contracts',
  },
];

const mockPrivacyViolations = [
  {
    id: '1',
    type: 'PII Exposure',
    severity: 'high',
    description: 'Email addresses found in unencrypted dataset',
    dataset: 'Security Dataset v2',
    count: 45,
    detectedAt: '2024-01-20 10:30:00',
    status: 'open',
    assignedTo: 'John Doe',
  },
  {
    id: '2',
    type: 'Data Retention Violation',
    severity: 'medium',
    description: 'Data retained beyond policy limits',
    dataset: 'Customer Dataset v1',
    count: 12,
    detectedAt: '2024-01-19 15:20:00',
    status: 'acknowledged',
    assignedTo: 'Jane Smith',
  },
  {
    id: '3',
    type: 'Access Control Issue',
    severity: 'high',
    description: 'Unauthorized access to sensitive data',
    dataset: 'Health Dataset v3',
    count: 3,
    detectedAt: '2024-01-18 09:45:00',
    status: 'resolved',
    assignedTo: 'Mike Johnson',
  },
];

const mockComplianceStatus = [
  {
    regulation: 'GDPR',
    status: 'compliant',
    score: 95,
    lastAudit: '2024-01-15',
    nextAudit: '2024-04-15',
    violations: 2,
    requirements: 45,
    completed: 43,
  },
  {
    regulation: 'CCPA',
    status: 'compliant',
    score: 92,
    lastAudit: '2024-01-10',
    nextAudit: '2024-04-10',
    violations: 1,
    requirements: 28,
    completed: 27,
  },
  {
    regulation: 'HIPAA',
    status: 'partial',
    score: 78,
    lastAudit: '2024-01-05',
    nextAudit: '2024-02-05',
    violations: 5,
    requirements: 32,
    completed: 25,
  },
  {
    regulation: 'SOX',
    status: 'non-compliant',
    score: 65,
    lastAudit: '2024-01-01',
    nextAudit: '2024-02-01',
    violations: 8,
    requirements: 18,
    completed: 12,
  },
];

const mockAnonymizationJobs = [
  {
    id: '1',
    name: 'Customer Data Anonymization',
    dataset: 'Customer Dataset v1',
    status: 'running',
    progress: 65,
    method: 'k-anonymity',
    startTime: '2024-01-20 10:00:00',
    estimatedEnd: '2024-01-20 16:00:00',
    recordsProcessed: 65000,
    totalRecords: 100000,
  },
  {
    id: '2',
    name: 'Health Data Pseudonymization',
    dataset: 'Health Dataset v2',
    status: 'completed',
    progress: 100,
    method: 'differential privacy',
    startTime: '2024-01-19 08:00:00',
    endTime: '2024-01-19 14:30:00',
    recordsProcessed: 25000,
    totalRecords: 25000,
  },
  {
    id: '3',
    name: 'Financial Data Masking',
    dataset: 'Financial Dataset v1',
    status: 'scheduled',
    progress: 0,
    method: 'tokenization',
    startTime: '2024-01-21 09:00:00',
    estimatedEnd: '2024-01-21 12:00:00',
    recordsProcessed: 0,
    totalRecords: 50000,
  },
];

const mockPrivacyPolicies = [
  {
    id: '1',
    name: 'Data Retention Policy',
    version: '2.1',
    status: 'active',
    lastUpdated: '2024-01-15',
    nextReview: '2024-04-15',
    category: 'Data Management',
    description: 'Defines how long different types of data should be retained',
  },
  {
    id: '2',
    name: 'Data Access Policy',
    version: '1.8',
    status: 'active',
    lastUpdated: '2024-01-10',
    nextReview: '2024-03-10',
    category: 'Access Control',
    description: 'Governs who can access what data and under what conditions',
  },
  {
    id: '3',
    name: 'Data Anonymization Policy',
    version: '1.5',
    status: 'draft',
    lastUpdated: '2024-01-20',
    nextReview: '2024-02-20',
    category: 'Privacy Protection',
    description: 'Standards for anonymizing and pseudonymizing data',
  },
];

const DataPrivacy: React.FC = () => {
  const [value, setValue] = useState(0);
  const [policyDialogOpen, setPolicyDialogOpen] = useState(false);
  const [anonymizationDialogOpen, setAnonymizationDialogOpen] = useState(false);
  const [selectedPolicy, setSelectedPolicy] = useState('');
  const [policyName, setPolicyName] = useState('');
  const [policyDescription, setPolicyDescription] = useState('');
  const [policyCategory, setPolicyCategory] = useState('');

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const handleCreatePolicy = () => {
    console.log('Creating privacy policy:', { policyName, policyDescription, policyCategory });
    setPolicyDialogOpen(false);
    setPolicyName('');
    setPolicyDescription('');
    setPolicyCategory('');
  };

  const handleStartAnonymization = () => {
    console.log('Starting anonymization job');
    setAnonymizationDialogOpen(false);
  };

  const getSensitivityColor = (sensitivity: string) => {
    switch (sensitivity) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      case 'low':
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant':
        return 'success';
      case 'partial':
        return 'warning';
      case 'non-compliant':
        return 'error';
      case 'active':
        return 'success';
      case 'draft':
        return 'warning';
      case 'running':
        return 'info';
      case 'completed':
        return 'success';
      case 'scheduled':
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
            Data Privacy & Compliance
          </Typography>
          <Typography variant="body1" color="text.secondary">
            GDPR compliance, data classification, and anonymization
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button variant="outlined" startIcon={<Policy />} onClick={() => setPolicyDialogOpen(true)}>
            Manage Policies
          </Button>
          <Button variant="contained" startIcon={<Add />} onClick={() => setAnonymizationDialogOpen(true)}>
            Start Anonymization
          </Button>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Privacy Violations"
            value={mockPrivacyViolations.length}
            icon={<Warning />}
            color="#f44336"
            trend={-2}
            subtitle="active alerts"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Compliance Score"
            value="87%"
            icon={<Shield />}
            color="#4caf50"
            trend={5.2}
            subtitle="average across regulations"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Data Classified"
            value="4,690"
            icon={<DataObject />}
            color="#2196f3"
            trend={12.5}
            subtitle="data points"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Anonymization Jobs"
            value={mockAnonymizationJobs.length}
            icon={<PrivacyTip />}
            color="#ff9800"
            trend={1}
            subtitle="active jobs"
          />
        </Grid>
      </Grid>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={value} onChange={handleChange} aria-label="data privacy tabs">
            <Tab icon={<DataObject />} label="Data Classification" />
            <Tab icon={<Warning />} label="Privacy Violations" />
            <Tab icon={<Gavel />} label="Compliance" />
            <Tab icon={<PrivacyTip />} label="Anonymization" />
            <Tab icon={<Policy />} label="Policies" />
          </Tabs>
        </Box>
        
        <TabPanel value={value} index={0}>
          <Typography variant="h6" gutterBottom>
            Data Classification & Sensitivity
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor and manage data classification based on sensitivity and privacy requirements
          </Typography>

          <Grid container spacing={3}>
            {mockDataClassification.map((classification) => (
              <Grid item xs={12} md={6} lg={3} key={classification.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {classification.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {classification.category}
                        </Typography>
                      </Box>
                      <Chip
                        label={classification.sensitivity.toUpperCase()}
                        color={getSensitivityColor(classification.sensitivity) as any}
                        size="small"
                      />
                    </Box>

                    <Typography variant="body2" color="text.secondary" mb={2}>
                      {classification.description}
                    </Typography>

                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {classification.count.toLocaleString()}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Records
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {classification.status}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Status
                        </Typography>
                      </Box>
                    </Box>

                    <Typography variant="caption" color="text.secondary" mb={2}>
                      Last scanned: {classification.lastScanned}
                    </Typography>

                    <Box display="flex" gap={1} mt={2}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Visibility />}
                        fullWidth
                      >
                        View Details
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Edit />}
                        fullWidth
                      >
                        Edit
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
            Privacy Violations & Alerts
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Track and manage privacy violations and security incidents
          </Typography>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Type</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Description</TableCell>
                  <TableCell>Dataset</TableCell>
                  <TableCell>Count</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Assigned To</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {mockPrivacyViolations.map((violation) => (
                  <TableRow key={violation.id}>
                    <TableCell>
                      <Typography variant="subtitle2" fontWeight={600}>
                        {violation.type}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={violation.severity.toUpperCase()}
                        color={getSeverityColor(violation.severity) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {violation.description}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {violation.dataset}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {violation.count}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={violation.status.toUpperCase()}
                        color={getStatusColor(violation.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {violation.assignedTo}
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
                          <CheckCircle />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
        
        <TabPanel value={value} index={2}>
          <Typography variant="h6" gutterBottom>
            Compliance Monitoring
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Track compliance status across different regulations and standards
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Compliance Status Overview
                  </Typography>
                  <RechartsBarChart
                    data={mockComplianceStatus}
                    dataKey="score"
                    xAxisKey="regulation"
                    colors={['#4caf50', '#ff9800', '#f44336']}
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
                    Compliance Summary
                  </Typography>
                  <List>
                    {mockComplianceStatus.map((regulation, index) => (
                      <ListItem key={index} divider>
                        <ListItemIcon>
                          {regulation.status === 'compliant' ? (
                            <CheckCircle color="success" />
                          ) : regulation.status === 'partial' ? (
                            <Warning color="warning" />
                          ) : (
                            <Error color="error" />
                          )}
                        </ListItemIcon>
                        <ListItemText
                          primary={regulation.regulation}
                          secondary={`${regulation.score}% • ${regulation.completed}/${regulation.requirements} requirements`}
                        />
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
                    Detailed Compliance Status
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Regulation</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Score</TableCell>
                          <TableCell>Requirements</TableCell>
                          <TableCell>Violations</TableCell>
                          <TableCell>Last Audit</TableCell>
                          <TableCell>Next Audit</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {mockComplianceStatus.map((regulation, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Typography variant="subtitle2" fontWeight={600}>
                                {regulation.regulation}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={regulation.status.toUpperCase()}
                                color={getStatusColor(regulation.status) as any}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {regulation.score}%
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {regulation.completed}/{regulation.requirements}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="error">
                                {regulation.violations}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {regulation.lastAudit}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {regulation.nextAudit}
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
            Data Anonymization & Pseudonymization
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Manage data anonymization jobs and privacy protection techniques
          </Typography>

          <Grid container spacing={3}>
            {mockAnonymizationJobs.map((job) => (
              <Grid item xs={12} md={6} lg={4} key={job.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {job.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {job.dataset}
                        </Typography>
                      </Box>
                      <Chip
                        label={job.status.toUpperCase()}
                        color={getStatusColor(job.status) as any}
                        size="small"
                      />
                    </Box>

                    <Typography variant="body2" color="text.secondary" mb={2}>
                      Method: {job.method}
                    </Typography>

                    {job.status === 'running' && (
                      <Box mb={2}>
                        <Box display="flex" justifyContent="space-between" mb={1}>
                          <Typography variant="body2" color="text.secondary">
                            Progress
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {job.progress}%
                          </Typography>
                        </Box>
                        <LinearProgress variant="determinate" value={job.progress} />
                        <Box display="flex" justifyContent="space-between" mt={1}>
                          <Typography variant="caption" color="text.secondary">
                            {job.recordsProcessed.toLocaleString()}/{job.totalRecords.toLocaleString()} records
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {job.estimatedEnd}
                          </Typography>
                        </Box>
                      </Box>
                    )}

                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {job.recordsProcessed.toLocaleString()}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Processed
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {job.totalRecords.toLocaleString()}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Total
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {job.method}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Method
                        </Typography>
                      </Box>
                    </Box>

                    <Box display="flex" gap={1} mt={2}>
                      {job.status === 'running' && (
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Stop />}
                          fullWidth
                        >
                          Stop
                        </Button>
                      )}
                      {job.status === 'scheduled' && (
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<PlayArrow />}
                          fullWidth
                        >
                          Start Now
                        </Button>
                      )}
                      {job.status === 'completed' && (
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Visibility />}
                          fullWidth
                        >
                          View Results
                        </Button>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={4}>
          <Typography variant="h6" gutterBottom>
            Privacy Policies & Governance
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Manage privacy policies, governance frameworks, and compliance requirements
          </Typography>

          <Grid container spacing={3}>
            {mockPrivacyPolicies.map((policy) => (
              <Grid item xs={12} md={6} lg={4} key={policy.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {policy.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          v{policy.version} • {policy.category}
                        </Typography>
                      </Box>
                      <Chip
                        label={policy.status.toUpperCase()}
                        color={getStatusColor(policy.status) as any}
                        size="small"
                      />
                    </Box>

                    <Typography variant="body2" color="text.secondary" mb={2}>
                      {policy.description}
                    </Typography>

                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Box textAlign="center">
                        <Typography variant="body2" color="text.secondary">
                          Last Updated
                        </Typography>
                        <Typography variant="body2">
                          {policy.lastUpdated}
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="body2" color="text.secondary">
                          Next Review
                        </Typography>
                        <Typography variant="body2">
                          {policy.nextReview}
                        </Typography>
                      </Box>
                    </Box>

                    <Box display="flex" gap={1} mt={2}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Visibility />}
                        fullWidth
                      >
                        View
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Edit />}
                        fullWidth
                      >
                        Edit
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
      </Paper>

      {/* Policy Dialog */}
      <Dialog open={policyDialogOpen} onClose={() => setPolicyDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Privacy Policy</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Policy Name"
            value={policyName}
            onChange={(e) => setPolicyName(e.target.value)}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Description"
            value={policyDescription}
            onChange={(e) => setPolicyDescription(e.target.value)}
            margin="normal"
            multiline
            rows={3}
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Category</InputLabel>
            <Select value={policyCategory} onChange={(e) => setPolicyCategory(e.target.value)}>
              <MenuItem value="Data Management">Data Management</MenuItem>
              <MenuItem value="Access Control">Access Control</MenuItem>
              <MenuItem value="Privacy Protection">Privacy Protection</MenuItem>
              <MenuItem value="Data Retention">Data Retention</MenuItem>
              <MenuItem value="Data Sharing">Data Sharing</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPolicyDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleCreatePolicy}>
            Create Policy
          </Button>
        </DialogActions>
      </Dialog>

      {/* Anonymization Dialog */}
      <Dialog open={anonymizationDialogOpen} onClose={() => setAnonymizationDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Start Anonymization Job</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Dataset</InputLabel>
            <Select>
              <MenuItem value="customer-v1">Customer Dataset v1</MenuItem>
              <MenuItem value="health-v2">Health Dataset v2</MenuItem>
              <MenuItem value="financial-v1">Financial Dataset v1</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Anonymization Method</InputLabel>
            <Select>
              <MenuItem value="k-anonymity">K-Anonymity</MenuItem>
              <MenuItem value="differential-privacy">Differential Privacy</MenuItem>
              <MenuItem value="tokenization">Tokenization</MenuItem>
              <MenuItem value="masking">Data Masking</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth margin="normal">
            <InputLabel>Privacy Level</InputLabel>
            <Select>
              <MenuItem value="low">Low (k=2)</MenuItem>
              <MenuItem value="medium">Medium (k=5)</MenuItem>
              <MenuItem value="high">High (k=10)</MenuItem>
              <MenuItem value="custom">Custom</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAnonymizationDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleStartAnonymization}>
            Start Job
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DataPrivacy;