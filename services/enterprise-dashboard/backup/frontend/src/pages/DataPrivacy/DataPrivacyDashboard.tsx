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
  Snackbar
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
  Success as SuccessIcon
} from '@mui/icons-material';
import { useDataPrivacy } from '../../hooks/useDataPrivacy';
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
      id={`data-privacy-tabpanel-${index}`}
      aria-labelledby={`data-privacy-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const DataPrivacyDashboard: React.FC = () => {
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
  const [anonymizeDialogOpen, setAnonymizeDialogOpen] = useState(false);
  const [anonymizeText, setAnonymizeText] = useState('');
  const [anonymizeResult, setAnonymizeResult] = useState<any>(null);
  const [dataSubjectDialogOpen, setDataSubjectDialogOpen] = useState(false);
  const [newDataSubject, setNewDataSubject] = useState({
    subject_id: '',
    email: '',
    data_categories: [] as string[]
  });
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  // API Hooks
  const {
    compliance,
    auditLogs,
    dataSubjects,
    retentionPolicies,
    loading,
    error,
    refetch,
    execute
  } = useDataPrivacy(timeRange);

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
    console.log(`Exporting data privacy metrics in ${exportFormat} format`);
    setExportDialogOpen(false);
  };

  const handleAnonymize = async () => {
    if (!anonymizeText.trim()) return;
    
    try {
      // This would call the actual API
      const result = {
        original_text: anonymizeText,
        anonymized_text: anonymizeText.replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, '[EMAIL_REDACTED]')
          .replace(/\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b/g, '[PHONE_REDACTED]'),
        anonymization_method: 'regex_replacement',
        pii_detected: ['email', 'phone'],
        confidence_score: 0.95,
        anonymized_at: new Date().toISOString()
      };
      setAnonymizeResult(result);
      setSnackbarMessage('Text anonymized successfully');
      setSnackbarOpen(true);
    } catch (error) {
      setSnackbarMessage('Error anonymizing text');
      setSnackbarOpen(true);
    }
  };

  const handleRegisterDataSubject = async () => {
    if (!newDataSubject.subject_id.trim()) return;
    
    try {
      // This would call the actual API
      setSnackbarMessage('Data subject registered successfully');
      setSnackbarOpen(true);
      setDataSubjectDialogOpen(false);
      setNewDataSubject({ subject_id: '', email: '', data_categories: [] });
      refetch();
    } catch (error) {
      setSnackbarMessage('Error registering data subject');
      setSnackbarOpen(true);
    }
  };

  // Data processing functions
  const getComplianceColor = (compliant: boolean) => {
    return compliant ? 'success' : 'error';
  };

  const getViolationSeverity = (violation: string) => {
    if (violation.includes('Low consent rate')) return { color: 'warning', icon: <Warning /> };
    if (violation.includes('Insufficient audit logging')) return { color: 'error', icon: <Error /> };
    return { color: 'info', icon: <Info /> };
  };

  const getConsentStatus = (consent: boolean, withdrawn: boolean) => {
    if (withdrawn) return { label: 'Withdrawn', color: 'error' };
    if (consent) return { label: 'Given', color: 'success' };
    return { label: 'Not Given', color: 'warning' };
  };

  // Render functions
  const renderOverviewCards = () => {
    if (loading) return <LoadingSkeleton variant="metrics" count={4} />;

    const gdprCompliant = compliance?.gdpr_compliant || false;
    const dataSubjectsCount = compliance?.data_subjects_count || 0;
    const consentRate = compliance?.consent_rate || 0;
    const anonymizationRate = compliance?.anonymization_rate || 0;

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    GDPR Compliance
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {gdprCompliant ? '100%' : '0%'}
                  </Typography>
                </Box>
                <Gavel sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Data Subjects
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {dataSubjectsCount}
                  </Typography>
                </Box>
                <PersonAdd sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Consent Rate
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {(consentRate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <CheckCircle sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
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
                    Anonymization Rate
                  </Typography>
                  <Typography variant="h4" component="div" color="white">
                    {(anonymizationRate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <PrivacyTip sx={{ fontSize: 40, color: 'white', opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderComplianceStatus = () => {
    if (loading) return <LoadingSkeleton variant="chart" />;

    const violations = compliance?.violations || [];
    const complianceData = [
      { metric: 'GDPR Compliance', value: compliance?.gdpr_compliant ? 100 : 0 },
      { metric: 'Data Retention', value: compliance?.data_retention_compliance ? 100 : 0 },
      { metric: 'Consent Rate', value: (compliance?.consent_rate || 0) * 100 },
      { metric: 'Anonymization Rate', value: (compliance?.anonymization_rate || 0) * 100 }
    ];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Compliance Metrics
              </Typography>
              <BarChart 
                data={complianceData}
                title="Compliance Metrics"
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
                Compliance Status
              </Typography>
              
              <Box mb={3}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Overall Compliance</Typography>
                  <Chip 
                    label={compliance?.gdpr_compliant ? 'Compliant' : 'Non-Compliant'} 
                    color={getComplianceColor(compliance?.gdpr_compliant || false)}
                    variant="filled"
                  />
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={compliance?.gdpr_compliant ? 100 : 0}
                  color={getComplianceColor(compliance?.gdpr_compliant || false)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box mb={3}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Data Retention</Typography>
                  <Chip 
                    label={compliance?.data_retention_compliance ? 'Compliant' : 'Non-Compliant'} 
                    color={getComplianceColor(compliance?.data_retention_compliance || false)}
                    variant="outlined"
                  />
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={compliance?.data_retention_compliance ? 100 : 0}
                  color={getComplianceColor(compliance?.data_retention_compliance || false)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box mb={3}>
                <Typography variant="body2" gutterBottom>
                  Last Compliance Check
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {compliance?.last_compliance_check ? 
                    new Date(compliance.last_compliance_check).toLocaleString() : 
                    'Never'
                  }
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {violations.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h2" gutterBottom>
                  Compliance Violations
                </Typography>
                <List>
                  {violations.map((violation: string, index: number) => {
                    const severity = getViolationSeverity(violation);
                    return (
                      <ListItem key={index} divider>
                        <ListItemIcon>
                          <Chip 
                            icon={severity.icon}
                            label={violation}
                            color={severity.color as any}
                            variant="outlined"
                          />
                        </ListItemIcon>
                        <ListItemText 
                          primary={violation}
                          secondary="Requires immediate attention"
                        />
                        <Button size="small" variant="outlined" color={severity.color as any}>
                          Fix
                        </Button>
                      </ListItem>
                    );
                  })}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    );
  };

  const renderDataSubjects = () => {
    if (loading) return <LoadingSkeleton variant="table" count={5} />;

    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" component="h2">
              Data Subjects
            </Typography>
            <Button
              variant="contained"
              startIcon={<PersonAdd />}
              onClick={() => setDataSubjectDialogOpen(true)}
            >
              Register Subject
            </Button>
          </Box>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Subject ID</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Data Categories</TableCell>
                  <TableCell align="center">Consent Status</TableCell>
                  <TableCell align="right">Created</TableCell>
                  <TableCell align="right">Last Accessed</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {dataSubjects?.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((subject: any, index: number) => {
                  const consentStatus = getConsentStatus(subject.consent_given, subject.consent_withdrawn);
                  
                  return (
                    <TableRow key={index} hover>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: 'primary.main' }}>
                            <PersonAdd />
                          </Avatar>
                          <Typography variant="body2" fontWeight="medium">
                            {subject.subject_id}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {subject.email || 'N/A'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box display="flex" gap={0.5} flexWrap="wrap">
                          {subject.data_categories?.map((category: string, idx: number) => (
                            <Chip key={idx} label={category} size="small" variant="outlined" />
                          ))}
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Chip 
                          label={consentStatus.label} 
                          size="small" 
                          color={consentStatus.color as any}
                          variant="filled"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="textSecondary">
                          {new Date(subject.created_at).toLocaleDateString()}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="textSecondary">
                          {new Date(subject.last_accessed).toLocaleDateString()}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Box display="flex" gap={0.5}>
                          <Tooltip title="View Details">
                            <IconButton size="small">
                              <Visibility />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Withdraw Consent">
                            <IconButton size="small" color="warning">
                              <PersonRemove />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete Subject">
                            <IconButton size="small" color="error">
                              <Delete />
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
            count={dataSubjects?.length || 0}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>
    );
  };

  const renderAuditLogs = () => {
    if (loading) return <LoadingSkeleton variant="table" count={5} />;

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            Audit Logs
          </Typography>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>User ID</TableCell>
                  <TableCell>Action</TableCell>
                  <TableCell>Resource</TableCell>
                  <TableCell align="center">Compliance Required</TableCell>
                  <TableCell align="right">IP Address</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {auditLogs?.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((log: any, index: number) => (
                  <TableRow key={index} hover>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(log.timestamp).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {log.user_id || 'System'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={log.action} 
                        size="small" 
                        color="primary"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {log.resource}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={log.compliance_required ? 'Yes' : 'No'} 
                        size="small" 
                        color={log.compliance_required ? 'warning' : 'default'}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" color="textSecondary">
                        {log.ip_address || 'N/A'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={auditLogs?.length || 0}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>
    );
  };

  const renderAnonymizationTool = () => {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            Data Anonymization Tool
          </Typography>
          
          <Box mb={2}>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Text to Anonymize"
              value={anonymizeText}
              onChange={(e) => setAnonymizeText(e.target.value)}
              placeholder="Enter text containing PII (emails, phone numbers, etc.)..."
              variant="outlined"
            />
          </Box>
          
          <Box display="flex" gap={2} mb={2}>
            <Button
              variant="contained"
              startIcon={<PrivacyTip />}
              onClick={handleAnonymize}
              disabled={!anonymizeText.trim()}
            >
              Anonymize Text
            </Button>
            <Button
              variant="outlined"
              startIcon={<ContentCopy />}
              onClick={() => navigator.clipboard.writeText(anonymizeText)}
            >
              Copy Original
            </Button>
          </Box>
          
          {anonymizeResult && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Anonymization Result:
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {anonymizeResult.anonymized_text}
                </Typography>
              </Paper>
              
              <Box mt={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Detected PII:
                </Typography>
                <Box display="flex" gap={1} flexWrap="wrap">
                  {anonymizeResult.pii_detected?.map((pii: string, index: number) => (
                    <Chip key={index} label={pii} size="small" color="warning" variant="outlined" />
                  ))}
                </Box>
              </Box>
              
              <Box mt={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Confidence Score: {(anonymizeResult.confidence_score * 100).toFixed(1)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={anonymizeResult.confidence_score * 100}
                  color="primary"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </Box>
          )}
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
            Data Privacy Dashboard
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
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="data privacy tabs">
            <Tab label="Compliance Status" icon={<Gavel />} />
            <Tab label="Data Subjects" icon={<PersonAdd />} />
            <Tab label="Audit Logs" icon={<Timeline />} />
            <Tab label="Anonymization" icon={<PrivacyTip />} />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            {renderComplianceStatus()}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {renderDataSubjects()}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {renderAuditLogs()}
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {renderAnonymizationTool()}
          </TabPanel>
        </Box>

        {/* Data Subject Registration Dialog */}
        <Dialog open={dataSubjectDialogOpen} onClose={() => setDataSubjectDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Register Data Subject</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              label="Subject ID"
              value={newDataSubject.subject_id}
              onChange={(e) => setNewDataSubject({...newDataSubject, subject_id: e.target.value})}
              margin="normal"
              required
            />
            <TextField
              fullWidth
              label="Email"
              type="email"
              value={newDataSubject.email}
              onChange={(e) => setNewDataSubject({...newDataSubject, email: e.target.value})}
              margin="normal"
            />
            <TextField
              fullWidth
              label="Data Categories (comma-separated)"
              value={newDataSubject.data_categories.join(', ')}
              onChange={(e) => setNewDataSubject({
                ...newDataSubject, 
                data_categories: e.target.value.split(',').map(c => c.trim()).filter(c => c)
              })}
              margin="normal"
              placeholder="personal, contact, financial"
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDataSubjectDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleRegisterDataSubject} variant="contained">
              Register
            </Button>
          </DialogActions>
        </Dialog>

        {/* Export Dialog */}
        <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)}>
          <DialogTitle>Export Data Privacy Metrics</DialogTitle>
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

export default DataPrivacyDashboard;
